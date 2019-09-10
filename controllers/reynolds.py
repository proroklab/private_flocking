import airsim
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Controller(object):
    def __init__(self, drone, flock_list, args):
        self.drone = drone
        self.flock_list = flock_list
        self.pos2v_scale = args.pos2v_scale
        self.sep_weight = args.sep_weight
        self.ali_weight = args.ali_weight
        self.coh_weight = args.coh_weight
        self.sep_max_cutoff = args.sep_max_cutoff
        self.ali_radius = args.ali_radius
        self.coh_radius = args.coh_radius

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
                    count +=1
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
        v_sep = self.get_sep_velocity()
        v_ali = self.get_ali_velocity()
        v_coh = self.get_coh_velocity()
        v_desired = self.sep_weight*v_sep + self.ali_weight*v_ali + self.coh_weight*v_coh
        self.drone.move_by_velocity(v_desired)





