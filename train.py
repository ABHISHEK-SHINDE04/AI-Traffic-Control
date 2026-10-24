# from __future__ import absolute_import
# from __future__ import print_function

# import os
# import sys
# import time
# import optparse
# import random
# import serial
# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.nn as nn
# import matplotlib.pyplot as plt

# # we need to import python modules from the $SUMO_HOME/tools directory
# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

# from sumolib import checkBinary  # noqa
# import traci  # noqa

# def get_vehicle_numbers(lanes):
#     vehicle_per_lane = dict()
#     for l in lanes:
#         vehicle_per_lane[l] = 0
#         for k in traci.lane.getLastStepVehicleIDs(l):
#             if traci.vehicle.getLanePosition(k) > 10:
#                 vehicle_per_lane[l] += 1
#     return vehicle_per_lane


# def get_waiting_time(lanes):
#     waiting_time = 0
#     for lane in lanes:
#         waiting_time += traci.lane.getWaitingTime(lane)
#     return waiting_time


# def phaseDuration(junction, phase_time, phase_state):
#     traci.trafficlight.setRedYellowGreenState(junction, phase_state)
#     traci.trafficlight.setPhaseDuration(junction, phase_time)



# class Model(nn.Module):
#     def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
#         super(Model, self).__init__()
#         self.lr = lr
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions

#         self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
#         self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
#         self.loss = nn.MSELoss()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.to(self.device)

#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         actions = self.linear3(x)
#         return actions


# class Agent:
#     def __init__(
#         self,
#         gamma,
#         epsilon,
#         lr,
#         input_dims,
#         fc1_dims,
#         fc2_dims,
#         batch_size,
#         n_actions,
#         junctions,
#         max_memory_size=100000,
#         epsilon_dec=5e-4,
#         epsilon_end=0.05,
#     ):
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.lr = lr
#         self.batch_size = batch_size
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         self.action_space = [i for i in range(n_actions)]
#         self.junctions = junctions
#         self.max_mem = max_memory_size
#         self.epsilon_dec = epsilon_dec
#         self.epsilon_end = epsilon_end
#         self.mem_cntr = 0
#         self.iter_cntr = 0
#         self.replace_target = 100

#         self.Q_eval = Model(
#             self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
#         )
#         self.memory = dict()
#         for junction in junctions:
#             self.memory[junction] = {
#                 "state_memory": np.zeros(
#                     (self.max_mem, self.input_dims), dtype=np.float32
#                 ),
#                 "new_state_memory": np.zeros(
#                     (self.max_mem, self.input_dims), dtype=np.float32
#                 ),
#                 "reward_memory":np.zeros(self.max_mem, dtype=np.float32),
#                 "action_memory": np.zeros(self.max_mem, dtype=np.int32),
#                 "terminal_memory": np.zeros(self.max_mem, dtype=np.bool_),
#                 "mem_cntr": 0,
#                 "iter_cntr": 0,
#             }


#     def store_transition(self, state, state_, action,reward, done,junction):
#         index = self.memory[junction]["mem_cntr"] % self.max_mem
#         self.memory[junction]["state_memory"][index] = state
#         self.memory[junction]["new_state_memory"][index] = state_
#         self.memory[junction]['reward_memory'][index] = reward
#         self.memory[junction]['terminal_memory'][index] = done
#         self.memory[junction]["action_memory"][index] = action
#         self.memory[junction]["mem_cntr"] += 1

#     def choose_action(self, observation):
#         state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
#         if np.random.random() > self.epsilon:
#             actions = self.Q_eval.forward(state)
#             action = torch.argmax(actions).item()
#         else:
#             action = np.random.choice(self.action_space)
#         return action
    
#     def reset(self,junction_numbers):
#         for junction_number in junction_numbers:
#             self.memory[junction_number]['mem_cntr'] = 0

#     def save(self,model_name):
#         torch.save(self.Q_eval.state_dict(),f'models/{model_name}.bin')

#     def learn(self, junction):
#         self.Q_eval.optimizer.zero_grad()

#         batch= np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32)

#         state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(
#             self.Q_eval.device
#         )
#         new_state_batch = torch.tensor(
#             self.memory[junction]["new_state_memory"][batch]
#         ).to(self.Q_eval.device)
#         reward_batch = torch.tensor(
#             self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
#         terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
#         action_batch = self.memory[junction]["action_memory"][batch]

#         q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
#         q_next = self.Q_eval.forward(new_state_batch)
#         q_next[terminal_batch] = 0.0
#         q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
#         loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

#         loss.backward()
#         self.Q_eval.optimizer.step()

#         self.iter_cntr += 1
#         self.epsilon = (
#             self.epsilon - self.epsilon_dec
#             if self.epsilon > self.epsilon_end
#             else self.epsilon_end
#         )


# def run(train=True,model_name="model",epochs=50,steps=500,ard=False):
#     if ard:
#         arduino = serial.Serial(port="/dev/cu.usbmodem101", baudrate=9600, timeout=.1)
#         def write_read(x):
#             arduino.write(bytes(x, 'utf-8'))
#             time.sleep(0.05)
#             data = arduino.readline()
#             return data
#     """execute the TraCI control loop"""
#     epochs = epochs
#     steps = steps
#     best_time = np.inf
#     total_time_list = list()
#     traci.start(
#         [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
#     )
#     all_junctions = traci.trafficlight.getIDList()
#     junction_numbers = list(range(len(all_junctions)))

#     brain = Agent(
#         gamma=0.99,
#         epsilon=0.0,
#         lr=0.1,
#         input_dims=4,
#         # input_dims = len(all_junctions) * 4,
#         fc1_dims=256,
#         fc2_dims=256,
#         batch_size=1024,
#         n_actions=4,
#         junctions=junction_numbers,
#     )

#     if not train:
#         brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin',map_location=brain.Q_eval.device))

#     print(brain.Q_eval.device)
#     traci.close()
#     for e in range(epochs):
#         if train:
#             traci.start(
#             [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
#             )
#         else:
#             traci.start(
#             [checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
#             )

#         print(f"epoch: {e}")
#         select_lane = [
#             ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
#             ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
#             ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
#             ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
#         ]

#         # select_lane = [
#         #     ["yyyyrrrrrrrrrrrr", "GGGGrrrrrrrrrrrr"],
#         #     ["rrrryyyyrrrrrrrr", "rrrrGGGGrrrrrrrr"],
#         #     ["rrrrrrrryyyyrrrr", "rrrrrrrrGGGGrrrr"],
#         #     ["rrrrrrrrrrrryyyy", "rrrrrrrrrrrrGGGG"],
#         # ]

#         step = 0
#         total_time = 0
#         min_duration = 5
        
#         traffic_lights_time = dict()
#         prev_wait_time = dict()
#         prev_vehicles_per_lane = dict()
#         prev_action = dict()
#         all_lanes = list()
        
#         for junction_number, junction in enumerate(all_junctions):
#             prev_wait_time[junction] = 0
#             prev_action[junction_number] = 0
#             traffic_lights_time[junction] = 0
#             prev_vehicles_per_lane[junction_number] = [0] * 4
#             # prev_vehicles_per_lane[junction_number] = [0] * (len(all_junctions) * 4) 
#             all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

#         while step <= steps:
#             traci.simulationStep()
#             for junction_number, junction in enumerate(all_junctions):
#                 controled_lanes = traci.trafficlight.getControlledLanes(junction)
#                 waiting_time = get_waiting_time(controled_lanes)
#                 total_time += waiting_time
#                 if traffic_lights_time[junction] == 0:
#                     vehicles_per_lane = get_vehicle_numbers(controled_lanes)
#                     # vehicles_per_lane = get_vehicle_numbers(all_lanes)

#                     #storing previous state and current state
#                     reward = -1 *  waiting_time
#                     state_ = list(vehicles_per_lane.values()) 
#                     state = prev_vehicles_per_lane[junction_number]
#                     prev_vehicles_per_lane[junction_number] = state_
#                     brain.store_transition(state, state_, prev_action[junction_number],reward,(step==steps),junction_number)

#                     #selecting new action based on current state
#                     lane = brain.choose_action(state_)
#                     prev_action[junction_number] = lane
#                     phaseDuration(junction, 6, select_lane[lane][0])
#                     phaseDuration(junction, min_duration + 10, select_lane[lane][1])

#                     if ard:
#                         ph = str(traci.trafficlight.getRedYellowGreenState("gneJ2"))
#                         if ph=="GGGrrrrrrrrr":
#                             ph=0
#                         elif ph=="rrrGGGrrrrrr":
#                             ph=2
#                         elif ph=="rrrrrrGGGrrr":
#                             ph=4
#                         elif ph=="rrrrrrrrrGGG":
#                             ph=6
#                         value = write_read(str(ph))

#                     traffic_lights_time[junction] = min_duration + 10
#                     if train:
#                         brain.learn(junction_number)
#                 else:
#                     traffic_lights_time[junction] -= 1
#             step += 1
#         print("total_time",total_time)
#         total_time_list.append(total_time)

#         if total_time < best_time:
#             best_time = total_time
#             if train:
#                 brain.save(model_name)

#         traci.close()
#         sys.stdout.flush()
#         if not train:
#             break
#     if train:
#         plt.plot(list(range(len(total_time_list))),total_time_list)
#         plt.xlabel("epochs")
#         plt.ylabel("total time")
#         plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
#         plt.show()

# def get_options():
#     optParser = optparse.OptionParser()
#     optParser.add_option(
#         "-m",
#         dest='model_city1',
#         type='string',
#         default="model",
#         help="name of model",
#     )
#     optParser.add_option(
#         "--train",
#         action = 'store_true',
#         default=False,
#         help="training or testing",
#     )
#     optParser.add_option(
#         "-e",
#         dest='epochs',
#         type='int',
#         default=50,
#         help="Number of epochs",
#     )
#     optParser.add_option(
#         "-s",
#         dest='steps',
#         type='int',
#         default=500,
#         help="Number of steps",
#     )
#     optParser.add_option(
#        "--ard",
#         action='store_true',
#         default=False,
#         help="Connect Arduino", 
#     )
#     options, args = optParser.parse_args()
#     return options


# # this is the main entry point of this script
# if __name__ == "__main__":
#     options = get_options()
#     model_name = options.model_city1
#     train = options.train
#     epochs = options.epochs
#     steps = options.steps
#     ard = options.ard
#     run(train=train,model_name=model_name,epochs=epochs,steps=steps,ard=ard)



from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# SUMO PATH 
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("ERROR: please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci



# HELPER FUNCTIONS

# Vehicle count per lane
def get_vehicle_numbers(lanes):
    vehicle_per_lane = []

    for l in lanes:
        count = 0

        for v in traci.lane.getLastStepVehicleIDs(l):

            # Ignore vehicles near junction
            if traci.vehicle.getLanePosition(v) > 10:
                count += 1

        vehicle_per_lane.append(count)

    return vehicle_per_lane


# Real waiting time
def get_waiting_time(lanes):

    total_wait = 0

    for l in lanes:

        vehicles = traci.lane.getLastStepVehicleIDs(l)

        for v in vehicles:
            total_wait += traci.vehicle.getWaitingTime(v)

    return total_wait



# SMART VEHICLE CONTROL

# Slow vehicles near traffic
def control_vehicle_speed(lanes):

    for lane in lanes:

        vehicles = traci.lane.getLastStepVehicleIDs(lane)

        lane_count = len(vehicles)

        for v in vehicles:

            try:

                speed = traci.vehicle.getSpeed(v)

                # Heavy traffic -> slow cars
                if lane_count > 10:
                    traci.vehicle.slowDown(v, 2, 5)

                # Medium traffic
                elif lane_count > 5:
                    traci.vehicle.slowDown(v, 2, 3)

                # Empty road -> allow fast speed
                else:
                    max_speed = traci.lane.getMaxSpeed(lane)
                    traci.vehicle.setSpeed(v, max_speed)

            except:
                pass


# Find best traffic phase
def choose_best_phase(junction, phases):

    controlled_lanes = traci.trafficlight.getControlledLanes(junction)

    lane_vehicle_data = {}

    # Count traffic per lane
    for lane in controlled_lanes:
        lane_vehicle_data[lane] = traci.lane.getLastStepVehicleNumber(lane)

    best_phase = 0
    best_score = -1

    # Check every traffic signal phase
    for idx, phase in enumerate(phases):

        score = 0

        for lane_index, signal in enumerate(phase):

            if lane_index >= len(controlled_lanes):
                continue

            lane = controlled_lanes[lane_index]

            # Green lane gets priority
            if signal == 'G' or signal == 'g':
                score += lane_vehicle_data[lane]

                # Extra priority for empty opposite roads
                if lane_vehicle_data[lane] == 0:
                    score += 2

        if score > best_score:
            best_score = score
            best_phase = idx

    return best_phase



# MODEL
class Model(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):

        super().__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.to(self.device)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


# AGENT
class Agent:

    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
        max_memory=100000,
        epsilon_dec=5e-4,
        epsilon_end=0.05
    ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.action_space = list(range(n_actions))
        self.junctions = junctions

        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.max_memory = max_memory

        self.Q_eval = Model(
            lr,
            input_dims,
            fc1_dims,
            fc2_dims,
            n_actions
        )

        self.memory = {}

        for j in junctions:

            self.memory[j] = {

                "state": np.zeros(
                    (max_memory, input_dims),
                    dtype=np.float32
                ),

                "new_state": np.zeros(
                    (max_memory, input_dims),
                    dtype=np.float32
                ),

                "reward": np.zeros(
                    max_memory,
                    dtype=np.float32
                ),

                "action": np.zeros(
                    max_memory,
                    dtype=np.int32
                ),

                "done": np.zeros(
                    max_memory,
                    dtype=np.bool_
                ),

                "counter": 0
            }

    def store(self, s, s_, a, r, d, j):

        idx = self.memory[j]["counter"] % self.max_memory

        self.memory[j]["state"][idx] = s
        self.memory[j]["new_state"][idx] = s_
        self.memory[j]["reward"][idx] = r
        self.memory[j]["action"][idx] = a
        self.memory[j]["done"][idx] = d

        self.memory[j]["counter"] += 1

    def choose_action(self, obs):

        # AI Decision
        if np.random.random() > self.epsilon:

            state = torch.tensor(
                [obs],
                dtype=torch.float
            ).to(self.Q_eval.device)

            return torch.argmax(
                self.Q_eval(state)
            ).item()

        return np.random.choice(self.action_space)

    def learn(self, j):

        mem = self.memory[j]

        size = mem["counter"]

        if size < self.batch_size:
            return

        batch = np.random.choice(
            size,
            self.batch_size,
            replace=False
        )

        s = torch.tensor(
            mem["state"][batch]
        ).to(self.Q_eval.device)

        s_ = torch.tensor(
            mem["new_state"][batch]
        ).to(self.Q_eval.device)

        r = torch.tensor(
            mem["reward"][batch]
        ).to(self.Q_eval.device)

        d = torch.tensor(
            mem["done"][batch]
        ).to(self.Q_eval.device)

        a = mem["action"][batch]

        q_eval = self.Q_eval(s)[
            range(self.batch_size),
            a
        ]

        q_next = self.Q_eval(s_)

        q_next[d] = 0.0

        q_target = r + self.gamma * torch.max(
            q_next,
            dim=1
        )[0]

        loss = self.Q_eval.loss(
            q_target,
            q_eval
        )

        self.Q_eval.optimizer.zero_grad()

        loss.backward()

        self.Q_eval.optimizer.step()

        self.epsilon = max(
            self.epsilon - self.epsilon_dec,
            self.epsilon_end
        )

    def save(self, name):

        os.makedirs("models", exist_ok=True)

        torch.save(
            self.Q_eval.state_dict(),
            f"models/{name}.bin"
        )



# MAIN RUN

def run(
    train=True,
    model_name="model",
    epochs=30,
    steps=1500
):

    best_time = np.inf
    total_times = []

    for epoch in range(epochs):

        traci.start([
            checkBinary("sumo-gui"),
            "-c", "configuration.sumocfg",
            "--gui-settings-file", "gui-settings.xml",
            "--start",
            "--quit-on-end"
        ])

        print(f"\n===== EPOCH {epoch} =====")

        junctions = traci.trafficlight.getIDList()

        junction_ids = list(range(len(junctions)))

        sample_lanes = traci.trafficlight.getControlledLanes(
            junctions[0]
        )

        state_size = len(sample_lanes)

        print("Detected State Size:", state_size)

        logic = traci.trafficlight.getAllProgramLogics(
            junctions[0]
        )[0]

        phases = [p.state for p in logic.phases]

        n_actions = len(phases)

        print("Detected Actions:", n_actions)

        agent = Agent(
            gamma=0.99,
            epsilon=0.2,
            lr=0.001,
            input_dims=state_size,
            fc1_dims=256,
            fc2_dims=256,
            batch_size=64,
            n_actions=n_actions,
            junctions=junction_ids
        )

        traffic_timer = {
            j: 0 for j in junctions
        }

        prev_state = {
            i: [0] * state_size
            for i in junction_ids
        }

        prev_action = {
            i: 0 for i in junction_ids
        }

        prev_wait = {
            j: 0 for j in junctions
        }

        step = 0
        total_wait = 0

        while step <= steps:

            traci.simulationStep()

            for jid, j in enumerate(junctions):

                lanes = traci.trafficlight.getControlledLanes(j)

                # SMART SPEED CONTROL
                control_vehicle_speed(lanes)

                wait = get_waiting_time(lanes)

                wait = wait / 10.0

                total_wait += wait

                if traffic_timer[j] == 0:

                    state = get_vehicle_numbers(lanes)

                    reward = prev_wait[j] - wait

                    reward = max(min(reward, 10), -10)

                    prev_wait[j] = wait

                    agent.store(
                        prev_state[jid],
                        state,
                        prev_action[jid],
                        reward,
                        step == steps,
                        jid
                    )

                    # AI + SMART TRAFFIC PRIORITY
                   

                    ai_action = agent.choose_action(state)

                    smart_action = choose_best_phase(
                        j,
                        phases
                    )

                    # If traffic is high -> use smart action
                    if max(state) > 8:
                        action = smart_action
                    else:
                        action = ai_action

                    
                    max_traffic = max(state)

                    # Heavy traffic
                    if max_traffic > 15:
                        green_time = 40

                    # Medium traffic
                    elif max_traffic > 8:
                        green_time = 30

                    # Empty road
                    elif max_traffic == 0:
                        green_time = 10

                    else:
                        green_time = 20

                    
                    traci.trafficlight.setRedYellowGreenState(
                        j,
                        phases[action]
                    )

                    traci.trafficlight.setPhaseDuration(
                        j,
                        green_time
                    )

                    prev_state[jid] = state
                    prev_action[jid] = action

                    traffic_timer[j] = green_time

                    if train:
                        agent.learn(jid)

                else:
                    traffic_timer[j] -= 1

            step += 1

        print("TOTAL WAIT TIME:", total_wait)

        total_times.append(total_wait)

        if train and total_wait < best_time:

            best_time = total_wait

            agent.save(model_name)

        traci.close()

    if train:

        os.makedirs("plots", exist_ok=True)

        plt.plot(total_times)

        plt.xlabel("Epochs")

        plt.ylabel("Total Waiting Time")

        plt.title("Traffic Optimization")

        plt.savefig(
            f"plots/time_vs_epoch_{model_name}.png"
        )

        plt.show()



def get_options():

    opt = optparse.OptionParser()

    opt.add_option(
        "--train",
        action="store_true",
        default=False
    )

    opt.add_option(
        "-m",
        dest="model",
        default="model"
    )

    opt.add_option(
        "-e",
        dest="epochs",
        type="int",
        default=30
    )

    opt.add_option(
        "-s",
        dest="steps",
        type="int",
        default=1500
    )

    return opt.parse_args()[0]



if __name__ == "__main__":

    options = get_options()

    run(
        train=options.train,
        model_name=options.model,
        epochs=options.epochs,
        steps=options.steps
    )