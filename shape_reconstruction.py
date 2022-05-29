import os
import json
import importlib
from argparse import ArgumentParser

import numpy as np
import networkx as nx
import scipy.linalg as la
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.spatial import distance


class ShapeReconstruction:

    def __init__(self, cfg, data_cfg):
        self.cfg = cfg
        self.data_cfg = data_cfg

        # initialize graph
        self.G = nx.Graph()
        self.G.graph['rods'] = [(u, v) for (u, v) in self.data_cfg['color_to_rod'].values()]
        self.G.graph['tendons'] = [(u, v) for (u, v) in self.data_cfg['sensor_to_tendon'].values()]

        # add end cap (marker) node to graph
        for node, color in enumerate(self.data_cfg['node_to_color']):
            self.G.add_node(node)
            self.G.nodes[node]['color'] = color
            self.G.nodes[node]['pos_list'] = []

        # add tendon edge to graph
        for sensor_id, (u, v) in self.data_cfg['sensor_to_tendon'].items():
            self.G.add_edge(u, v)
            self.G.edges[u, v]['sensor_id'] = sensor_id
            self.G.edges[u, v]['type'] = 'tendon'

        # add rod edge to graph
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            self.G.add_edge(u, v)
            self.G.edges[u, v]['type'] = 'rod'
            self.G.edges[u, v]['length'] = self.data_cfg['rod_length']
            self.G.edges[u, v]['color'] = color
            self.G.edges[u, v]['pose_list'] = []


    def constrained_optimization(self, info):
        for u, v in self.data_cfg['color_to_rod'].values():
            self.G.nodes[u]['pos_list'].append(np.random.random(3))
            self.G.nodes[v]['pos_list'].append(np.random.random(3))

        rod_length = self.data_cfg['rod_length']
        num_endcaps = 2 * self.data_cfg['num_rods']  # a rod has two end caps

        sensor_measurement = info['sensors']
        obj_func, jac_func = self.objective_function_generator(sensor_measurement)

        init_values = np.zeros(3 * num_endcaps)
        for i in range(num_endcaps):
            init_values[(3 * i):(3 * i + 3)] = self.G.nodes[i]['pos_list'][-1]

        constraints = []
        for _, (u, v) in self.data_cfg['color_to_rod'].items():
            constraint = dict()
            constraint['type'] = 'eq'
            constraint['fun'], constraint['jac'] = self.endcap_constraint_generator(u, v, rod_length)
            constraints.append(constraint)

        res = minimize(obj_func, init_values, jac=jac_func, method='SLSQP', constraints=constraints)
        for i in range(num_endcaps):
            self.G.nodes[i]['pos_list'][-1] = res.x[(3 * i):(3 * i + 3)].copy()


    def objective_function_generator(self, sensor_measurement):

        def objective_function(X):
            binary_loss = 0
            for sensor_id, (u, v) in self.data_cfg['sensor_to_tendon'].items():
                sensor_id = str(sensor_id)
                if not self.data_cfg["sensor_status"][sensor_id] or sensor_measurement[sensor_id]['length'] <= 0:
                    continue

                u_pos = X[(3 * u):(3 * u + 3)]
                v_pos = X[(3 * v):(3 * v + 3)]
                estimated_length = la.norm(u_pos - v_pos)
                measured_length = sensor_measurement[sensor_id]['length'] / self.data_cfg['cable_length_scale']
                binary_loss += (estimated_length - measured_length)**2
            return binary_loss

        def jacobian_function(X):
            result = np.zeros_like(X)
            for sensor_id, (u, v) in self.data_cfg['sensor_to_tendon'].items():
                sensor_id = str(sensor_id)
                if not self.data_cfg['sensor_status'][sensor_id] or sensor_measurement[sensor_id]['length'] <= 0:
                    continue
                u_pos = X[(3 * u):(3 * u + 3)]
                v_pos = X[(3 * v):(3 * v + 3)]
                l_est = la.norm(u_pos - v_pos)
                l_gt = sensor_measurement[sensor_id]['length'] / self.data_cfg['cable_length_scale']
                result[3*u : 3*u + 3] += 2*(l_est - l_gt)*(u_pos - v_pos) / l_est
                result[3*v : 3*v + 3] += 2*(l_est - l_gt)*(v_pos - u_pos) / l_est
            return result

        return objective_function, jacobian_function


    def endcap_constraint_generator(self, u, v, rod_length):

        def constraint_function(X):
            u_pos = X[3 * u: 3 * u + 3]
            v_pos = X[3 * v: 3 * v + 3]
            return la.norm(u_pos - v_pos) - rod_length

        def jacobian_function(X):
            u_pos = X[3 * u : 3 * u + 3]
            v_pos = X[3 * v : 3 * v + 3]
            l_est = la.norm(u_pos - v_pos)  # denominator
            result = np.zeros_like(X)
            result[3 * u : 3 * u + 3] = (u_pos - v_pos) / l_est
            result[3 * v : 3 * v + 3] = (v_pos - u_pos) / l_est
            return result

        return constraint_function, jacobian_function


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--videos", default=[f"{i:04d}" for i in range(1, 17)], nargs="+")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=1000, type=int)
    parser.add_argument("--mocap_scale", default=1000, help="mm")
    args = parser.parse_args()

    error_list = []

    for video in args.videos:
        video_path = os.path.join(args.dataset, video)
        prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'color'))])
        end_frame = min(len(prefixes), args.end_frame)

        # read data config
        data_cfg_module = importlib.import_module(f'{args.dataset}.{video}.config')
        data_cfg = data_cfg_module.get_config(read_cfg=True)

        # initialize tracker
        rec = ShapeReconstruction(args, data_cfg)

        data = dict()
        for idx in range(args.start_frame, end_frame):
            prefix = prefixes[idx]
            info_path = os.path.join(video_path, 'data', f'{prefix}.json')
            data[prefix] = dict()

            with open(info_path, 'r') as f:
                info = json.load(f)
            data[prefix]['info'] = info


        for idx in tqdm(range(args.start_frame, end_frame)):
            prefix = prefixes[idx]
            info = data[prefix]['info']
            info['prefix'] = prefix
            rec.constrained_optimization(info)

            for u, v in rec.data_cfg['color_to_rod'].values():
                u_pos = rec.G.nodes[u]['pos_list'][-1]
                v_pos = rec.G.nodes[v]['pos_list'][-1]
                # print(distance.euclidean(u_pos, v_pos))

            for sensor, (u, v) in rec.data_cfg['sensor_to_tendon'].items():
                u_pos = rec.G.nodes[u]['pos_list'][-1]
                v_pos = rec.G.nodes[v]['pos_list'][-1]
                pred_length = distance.euclidean(u_pos, v_pos)
                # print(f"{sensor = }, dist: {pred_length}")

                measured_length = info['sensors'][sensor]['length'] / rec.data_cfg['cable_length_scale']

                u_pos = info['mocap'][str(u)]
                v_pos = info['mocap'][str(v)]
                if u_pos is not None and v_pos is not None:
                    u_pos = np.array([u_pos['x'], u_pos['y'], u_pos['z']]) / args.mocap_scale
                    v_pos = np.array([v_pos['x'], v_pos['y'], v_pos['z']]) / args.mocap_scale
                    gt_length = la.norm(u_pos - v_pos)
                    if 0 < gt_length < 0.5:
                        error = abs(pred_length - gt_length)
                        error_list.append(error)

    errors = np.array(error_list)
    print(f"{errors.mean() = }")
    print(f"{errors.std() = }")
