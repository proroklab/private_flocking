import numpy as np
import math

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Leader_Controller(object):
    def __init__(self, drone, flock_list, args):
        self.drone = drone
        self.flock_list = flock_list
        self.trajectory = args.trajectory
        self.v_leader = args.v_leader
        self.index = -1
        self.stop = False

        # Flocking
        self.pos2v_scale = args.pos2v_scale
        self.sep_weight = args.leader_sep_weight
        self.ali_weight = args.leader_ali_weight
        self.coh_weight = args.leader_coh_weight
        self.sep_max_cutoff = args.leader_sep_max_cutoff
        self.ali_radius = args.leader_ali_radius
        self.coh_radius = args.leader_coh_radius


        if self.trajectory == 'Line' or self.trajectory == 'Sinusoidal':
            self.line_len = args.line_len
            self.sine_period_ratio = args.sine_period_ratio
            self.sine_width = args.sine_width
            self.point_list = np.array([[0.0, 0.0, 0.0], [self.line_len, 0.0, 0.0]])
            self.lookahead = args.lookahead
        elif self.trajectory == 'Zigzag':
            self.line_len = args.line_len
            self.zigzag_len = args.zigzag_len
            self.zigzag_width = args.zigzag_width
            n = math.floor(self.line_len/self.zigzag_len)
            self.point_list = np.zeros((3*n+1, 3))
            for i in range(n):
                self.point_list[3*i+1] = np.array([(4*i+1)*self.zigzag_len/4, -self.zigzag_width, 0.0])
                self.point_list[3*i+2] = np.array([(4*i+3)*self.zigzag_len/4, self.zigzag_width, 0.0])
                self.point_list[3*i+3] = np.array([(4*i+4)*self.zigzag_len/4, 0.0, 0.0])
            self.lookahead = args.lookahead
            self.init_start_point = self.point_list[0]
            self.overall_unit_vec = normalize(self.point_list[3] - self.init_start_point)
        self.start_point = self.point_list[0]
        self.end_point = self.point_list[1]
        self.length = np.linalg.norm(self.end_point - self.start_point)
        self.unit_vec = normalize(self.end_point - self.start_point)

    def get_sep_velocity(self):
        v_sep = 0

        neighbors = self.drone.get_neighbors(self.flock_list)
        neighbor_number = len(neighbors)
        if neighbor_number == 0:
            return 0
        else:
            count = 0
            for neighbor in neighbors:
                pos = self.drone.get_position()
                pos_neighbor = self.drone.get_neighbor_position(neighbor, self.flock_list[neighbor])
                diff = pos - pos_neighbor
                distance = np.linalg.norm(diff)
                if 0 < distance < self.sep_max_cutoff:
                    normalize(diff)
                    v_sep += diff/distance
                    count += 1
            if count == 0:
                return 0
            else:
                v_sep /= count
            v_sep = normalize(v_sep) * self.pos2v_scale
            return v_sep

    def get_avg_neighbor_velocity_dis(self, neighbor_radius):
        v_neighbor = 0
        neighbors_dis = self.drone.get_neighbors_dis(self.flock_list, neighbor_radius)
        neighbor_number = len(neighbors_dis)
        if neighbor_number == 0:
            return self.drone.get_velocity()
        else:
            for neighbor in neighbors_dis:
                v = self.drone.get_neighbor_velocity(neighbor)
                v_neighbor += v
            v_neighbor += self.drone.get_velocity()
            return v_neighbor/(neighbor_number+1)

    def get_ali_velocity(self):
        v_neighbor_avg = self.get_avg_neighbor_velocity_dis(self.ali_radius)
        v_align = v_neighbor_avg
        # v_align = normalize(v_align) * self.v_max
        return v_align

    def get_avg_neighbor_pos_dis(self, neighbor_radius):
        pos_neighbor = 0
        neighbors_dis = self.drone.get_neighbors_dis(self.flock_list, neighbor_radius)
        neighbor_number = len(neighbors_dis)
        if neighbor_number == 0:
            return 0
        else:
            for neighbor in neighbors_dis:
                pos = self.drone.get_neighbor_position(neighbor, self.flock_list[neighbor])
                pos_neighbor += pos
            return pos_neighbor / neighbor_number

    def get_coh_velocity(self):
        pos_neighbor_avg = self.get_avg_neighbor_pos_dis(self.coh_radius)
        pos = self.drone.get_position()
        pos_coh = pos_neighbor_avg - pos
        v_coh = normalize(pos_coh) * self.pos2v_scale
        return v_coh

    def step(self):
        vx = 0.0
        vy = 0.0
        vz = 0.0
        err = 0.0

        if self.index == -1:
            pos = self.drone.get_position()
            self.point_list += pos
            self.start_point = self.point_list[0]
            self.end_point = self.point_list[1]
            self.index += 1

        pos = self.drone.get_position()
        distance = np.dot((pos - self.start_point), self.unit_vec)

        err = math.sqrt(np.linalg.norm(pos - self.start_point) ** 2 - distance ** 2)

        if (distance + self.lookahead) > self.length:
            self.index += 1
            if (self.index + 1) == len(self.point_list):
                self.index -= 1
                self.stop = True
            self.start_point = self.point_list[self.index]
            self.end_point = self.point_list[(self.index+1) % len(self.point_list)]
            self.length = np.linalg.norm(self.end_point - self.start_point)
            self.unit_vec = normalize(self.end_point - self.start_point)
            distance = np.dot((pos - self.start_point), self.unit_vec)

        if not self.stop:
            pos_desired = self.start_point + self.unit_vec * (distance + self.lookahead)
            if self.trajectory == 'Sinusoidal':
                pos_desired[1] += self.sine_width * math.sin((distance + self.lookahead) / self.sine_period_ratio)
                err = math.fabs(pos[1] - self.start_point[1] - self.sine_width * math.sin((pos[0] - self.start_point[0]) / self.sine_period_ratio))

            vx = pos_desired[0] - pos[0]
            vy = pos_desired[1] - pos[1]
            vz = pos_desired[2] - pos[2]


        self.drone.track_err = err
        v_leader = normalize(np.array([vx, vy, vz])) * self.v_leader
        # print(v_leader)

        if self.sep_weight != 0.0:
            v_leader_sep = self.get_sep_velocity()
            v_leader += self.sep_weight * v_leader_sep
        if self.ali_weight != 0.0:
            v_leader_ali = self.get_ali_velocity()
            v_leader += self.ali_weight * v_leader_ali
        if self.coh_weight != 0.0:
            v_leader_coh = self.get_coh_velocity()
            v_leader += self.coh_weight * v_leader_coh

        self.drone.move_by_velocity(v_leader)
