"""
UAV Network Traffic Scheduling Solver
Optimizes traffic flow distribution across UAV mesh network
"""

import sys
import math
from typing import List, Tuple, Dict
from collections import defaultdict


class UAV:
    """Represents a UAV node in the network"""
    def __init__(self, x: int, y: int, peak_bandwidth: float, phase: int):
        self.x = x
        self.y = y
        self.peak_bandwidth = peak_bandwidth
        self.phase = phase
    
    def get_bandwidth(self, t: int) -> float:
        """Calculate bandwidth at time t using periodic function b(φ+t)"""
        effective_t = (self.phase + t) % 10
        
        if effective_t in [0, 1, 8, 9]:
            return 0.0
        elif effective_t in [2, 7]:
            return self.peak_bandwidth / 2.0
        else:  # 3, 4, 5, 6
            return self.peak_bandwidth


class Flow:
    """Represents a traffic flow"""
    def __init__(self, flow_id: int, src_x: int, src_y: int, start_time: int, 
                 total_size: int, m1: int, n1: int, m2: int, n2: int):
        self.flow_id = flow_id
        self.src_x = src_x
        self.src_y = src_y
        self.start_time = start_time
        self.total_size = total_size
        self.m1 = m1
        self.n1 = n1
        self.m2 = m2
        self.n2 = n2
        self.remaining = total_size
        

class ScheduleRecord:
    """Represents a scheduling record for a flow"""
    def __init__(self, t: int, x: int, y: int, rate: float):
        self.t = t
        self.x = x
        self.y = y
        self.rate = rate


class UAVScheduler:
    """Main scheduler for UAV network traffic"""
    
    def __init__(self, M: int, N: int, T: int):
        self.M = M
        self.N = N
        self.T = T
        self.uavs: Dict[Tuple[int, int], UAV] = {}
        self.flows: List[Flow] = []
        
    def add_uav(self, x: int, y: int, peak_bandwidth: float, phase: int):
        """Add a UAV to the network"""
        self.uavs[(x, y)] = UAV(x, y, peak_bandwidth, phase)
    
    def add_flow(self, flow_id: int, src_x: int, src_y: int, start_time: int,
                 total_size: int, m1: int, n1: int, m2: int, n2: int):
        """Add a flow to be scheduled"""
        flow = Flow(flow_id, src_x, src_y, start_time, total_size, m1, n1, m2, n2)
        self.flows.append(flow)
    
    def manhattan_distance(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Calculate Manhattan distance (number of hops)"""
        return abs(x1 - x2) + abs(y1 - y2)
    
    def select_best_landing_uav(self, flow: Flow, current_time: int) -> Tuple[int, int]:
        """
        Select the best landing UAV for a flow based on:
        - Available bandwidth at current time
        - Distance from source
        - Within the allowed landing range
        """
        best_uav = None
        best_score = -1
        
        for x in range(flow.m1, flow.m2 + 1):
            for y in range(flow.n1, flow.n2 + 1):
                if (x, y) not in self.uavs:
                    continue
                
                uav = self.uavs[(x, y)]
                bandwidth = uav.get_bandwidth(current_time)
                
                if bandwidth <= 0:
                    continue
                
                # Calculate distance
                dist = self.manhattan_distance(flow.src_x, flow.src_y, x, y)
                
                # Scoring: prioritize closer UAVs with higher bandwidth
                # Normalize distance by grid size
                max_dist = self.M + self.N
                dist_score = 1.0 - (dist / max_dist)
                bw_score = bandwidth / 1000.0  # Normalize bandwidth
                
                score = 0.6 * bw_score + 0.4 * dist_score
                
                if score > best_score:
                    best_score = score
                    best_uav = (x, y)
        
        return best_uav
    
    def schedule_flow(self, flow: Flow) -> List[ScheduleRecord]:
        """
        Schedule a single flow using greedy approach:
        - Start from flow.start_time
        - Select best available landing UAV at each time slot
        - Allocate maximum available bandwidth
        """
        records = []
        remaining = flow.total_size
        current_time = flow.start_time
        previous_landing = None
        
        while remaining > 0 and current_time < self.T:
            # Select best landing UAV
            landing_uav = self.select_best_landing_uav(flow, current_time)
            
            if landing_uav is None:
                current_time += 1
                continue
            
            uav = self.uavs[landing_uav]
            available_bandwidth = uav.get_bandwidth(current_time)
            
            # Allocate traffic: min of remaining traffic and available bandwidth
            allocated = min(remaining, available_bandwidth)
            
            if allocated > 0:
                records.append(ScheduleRecord(
                    current_time, 
                    landing_uav[0], 
                    landing_uav[1], 
                    allocated
                ))
                remaining -= allocated
                previous_landing = landing_uav
            
            current_time += 1
        
        return records
    
    def calculate_flow_score(self, flow: Flow, records: List[ScheduleRecord]) -> float:
        """
        Calculate score for a flow based on the scoring function:
        1. Total U2G Traffic Score (0.4 weight)
        2. Traffic Delay Score (0.2 weight)
        3. Transmission Distance Score (0.3 weight)
        4. Landing UAV Point Score (0.1 weight)
        """
        if not records or flow.total_size == 0:
            return 0.0
        
        total_transmitted = sum(r.rate for r in records)
        
        # 1. Total U2G Traffic Score
        u2g_score = min(total_transmitted / flow.total_size, 1.0)
        
        # 2. Traffic Delay Score
        TMAX = 10.0
        delay_score = 0.0
        for record in records:
            ti = record.t - flow.start_time
            si = record.rate
            delay_score += (TMAX / (ti + TMAX)) * (si / flow.total_size)
        
        # 3. Transmission Distance Score
        distance_score = 0.0
        ALPHA = 0.1
        for record in records:
            hops = self.manhattan_distance(flow.src_x, flow.src_y, record.x, record.y)
            si = record.rate
            distance_score += (si / flow.total_size) * (2 ** (-ALPHA * hops))
        
        # 4. Landing UAV Point Score
        landing_points = set()
        for record in records:
            landing_points.add((record.x, record.y))
        k = len(landing_points)
        point_score = 1.0 / k if k > 0 else 0.0
        
        # Total score
        total_score = 100 * (
            0.4 * u2g_score +
            0.2 * delay_score +
            0.3 * distance_score +
            0.1 * point_score
        )
        
        return total_score
    
    def solve(self) -> Dict[int, List[ScheduleRecord]]:
        """
        Solve the scheduling problem for all flows
        Returns a dictionary mapping flow_id to schedule records
        """
        solution = {}
        
        for flow in self.flows:
            records = self.schedule_flow(flow)
            solution[flow.flow_id] = records
        
        return solution
    
    def calculate_total_score(self, solution: Dict[int, List[ScheduleRecord]]) -> float:
        """Calculate weighted total score across all flows"""
        total_traffic = sum(flow.total_size for flow in self.flows)
        
        if total_traffic == 0:
            return 0.0
        
        weighted_score = 0.0
        for flow in self.flows:
            flow_score = self.calculate_flow_score(flow, solution[flow.flow_id])
            weight = flow.total_size / total_traffic
            weighted_score += weight * flow_score
        
        return weighted_score
    
    def print_solution(self, solution: Dict[int, List[ScheduleRecord]]):
        """Print solution in the required output format"""
        for flow in self.flows:
            records = solution[flow.flow_id]
            print(f"{flow.flow_id} {len(records)}")
            for record in records:
                print(f"{record.t} {record.x} {record.y} {record.rate:.6f}")


def read_input(file_path: str = None) -> UAVScheduler:
    """Read input from file or stdin"""
    lines = []
    
    if file_path:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        lines = [line.strip() for line in sys.stdin if line.strip()]
    
    # Parse first line: M N FN T
    M, N, FN, T = map(int, lines[0].split())
    scheduler = UAVScheduler(M, N, T)
    
    # Parse UAV information (M*N lines)
    line_idx = 1
    for i in range(M * N):
        parts = lines[line_idx].split()
        x, y = int(parts[0]), int(parts[1])
        B = float(parts[2])
        phi = int(parts[3])
        scheduler.add_uav(x, y, B, phi)
        line_idx += 1
    
    # Parse flow information (FN lines)
    for i in range(FN):
        parts = lines[line_idx].split()
        f, x, y, tst, size, m1, n1, m2, n2 = map(int, parts)
        scheduler.add_flow(f, x, y, tst, size, m1, n1, m2, n2)
        line_idx += 1
    
    return scheduler


def main():
    """Main entry point"""
    # Check if input file is provided as command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        scheduler = read_input(input_file)
    else:
        scheduler = read_input()
    
    # Solve the problem
    solution = scheduler.solve()
    
    # Print solution
    scheduler.print_solution(solution)
    
    # Print score to stderr for debugging (optional)
    total_score = scheduler.calculate_total_score(solution)
    print(f"# Total Score: {total_score:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
