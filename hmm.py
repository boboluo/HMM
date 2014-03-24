import math
import numpy as np

class HMM:

    def __init__(self, states, observations):
        self.states = states
        self.states_size = len(states)
        self.states_dict = HMM.extractFlagStr(states)
        self.observations = observations
        self.observations_size = len(observations)
        self.observations_dict = HMM.extractFlagStr(observations)
        self.initModelList()

    def initWithSetting(self, init, trans, emit):
        self.addModel(
            HMM.createModel(
                HMM.normalize(np.array(init)),
                HMM.normalize(np.array(trans)),
                HMM.normalize(np.array(emit)),
            )
        )

    def initWithData(self, states_list, observations_list):
        model = HMM.createModel(
            np.zeros(self.states_size,),
            np.zeros((self.states_size, self.states_size)),
            np.zeros((self.states_size, self.observations_size))
            )

        for i in range(len(states_list)):
            states = states_list[i]
            length = len(states)
            if 0 == length:
                continue
            for j in range(length - 1):
                model['init'][self.states_dict[states[j]]] += 1
                model['trans'][self.states_dict[states[j]]][self.states_dict[states[j + 1]]] += 1
                model['emit'][self.states_dict[states[j]]][self.observations_dict[observations_list[i][j]]] += 1
            model['init'][self.states_dict[states[-1]]] += 1
            model['emit'][self.states_dict[states[-1]]][self.observations_dict[observations_list[i][-1]]] += 1

        HMM.laplaceSmoothing(model['init'])
        HMM.laplaceSmoothing(model['trans'])
        HMM.laplaceSmoothing(model['emit'])

        HMM.normalize(model['init'])
        HMM.normalize(model['trans'])
        HMM.normalize(model['emit'])

        self.addModel(model)


    @staticmethod
    def createModel(init, trans, emit):
        model = {}
        model['init'] = init
        model['trans'] = trans
        model['emit'] = emit
        return model

    def initModelList(self):
        self.model_list = []
        self.model_list_index = -1

    def addModel(self, model):
        self.model_list.append(model)
        self.model_list_index += 1

    def getCurrentModel(self):
        return self.model_list[self.model_list_index]


    def calculateA(self, observations):
        model = self.getCurrentModel()
        length = len(observations)
        matrix_a = np.zeros((length, self.states_size))

        for state in range(self.states_size):
            matrix_a[0, state] = model['init'][state] * model['emit'][state][self.observations_dict[observations[0]]]
        HMM.normalize(matrix_a[0])
        for obs_idx in range(1, length):
            for to_state in range(self.states_size):
                for from_state in range(self.states_size):
                    matrix_a[obs_idx][to_state] += matrix_a[obs_idx - 1][from_state] * model['trans'][from_state][to_state] * model['emit'][to_state][self.observations_dict[observations[obs_idx]]]
            HMM.normalize(matrix_a[obs_idx])

        return matrix_a

    def calculateB(self, observations):
        model = self.getCurrentModel()
        length = len(observations)
        matrix_b = np.zeros((length, self.states_size))

        for state in range(self.states_size):
            matrix_b[length - 1][state] = 1.0
        HMM.normalize(matrix_b[length - 1])
        for obs_idx in range(length - 1, 0, -1):
            for from_state in range(self.states_size):
                for to_state in range(self.states_size):
                    matrix_b[obs_idx - 1][from_state] += matrix_b[obs_idx][to_state] * model['trans'][from_state][to_state] * model['emit'][to_state][self.observations_dict[observations[obs_idx]]]
            HMM.normalize(matrix_b[obs_idx - 1])

        return matrix_b

    def calculateObsProbMatrixs(self, observations):
        matrix_a = self.calculateA(observations)
        matrix_b = self.calculateB(observations)
        length = len(observations)
        matrix_c = np.zeros((length, self.states_size))
        for obs_idx in range(length):
            for state_idx in range(self.states_size):
                matrix_c[obs_idx][state_idx] = matrix_a[obs_idx][state_idx] * matrix_b[obs_idx][state_idx]
        HMM.normalize(matrix_c)
        return matrix_a, matrix_b, matrix_c


    def filter(self, observations):
        pass

    def smooth(self, observations):
        pass

    def predict(self, observations, step = 1):
        model = self.getCurrentModel()
        prob = self.calculateA(observations)[-1]
        for s in range(step):
            prob_next = np.zeros(self.states_size)
            for i in range(self.states_size):
                for j in range(self.states_size):
                    prob_next[i] += prob[j] * model['trans'][j][i]
            HMM.normalize(prob_next)
            prob = prob_next
        return self.states[np.argmax(prob)]

    def decode(self, observations):
        model = self.getCurrentModel()
        state_prob = []
        state_decode = []
        for state_idx in range(self.states_size):
            state_prob.append(model['init'][state_idx] * model['emit'][state_idx][self.observations_dict[observations[0]]])
            state_decode.append(self.states[state_idx])
        HMM.normalize(state_prob)
        for each_obs in observations[1:]:
            state_prob_next = []
            state_decode_next = []
            for to_state in range(self.states_size):
                prob = 0
                pre = 0
                for from_state in range(self.states_size):
                    prob_tmp = state_prob[from_state] * model['trans'][from_state][to_state]
                    if prob_tmp > prob:
                        prob = prob_tmp
                        pre = from_state
                state_prob_next.append(prob * model['emit'][to_state][self.observations_dict[each_obs]])

                state_decode_next.append(state_decode[pre] + self.states[to_state])
            state_prob = state_prob_next
            state_decode = state_decode_next
            HMM.normalize(state_prob)
        return state_decode[np.argmax(state_prob)]


    def polish(self, observations):
        model = self.getCurrentModel()
        matrix_a, matrix_b, matrix_c = self.calculateObsProbMatrixs(observations)
        length = len(observations)
        state_trans_cube = np.zeros((length - 1, self.states_size, self.states_size))
        for obs_idx in range(length - 1):
            for from_state in range(self.states_size):
                for to_state in range(self.states_size):
                    state_trans_cube[obs_idx][from_state][to_state] = matrix_a[obs_idx][from_state] * model['trans'][from_state][to_state] * model['emit'][to_state][self.observations_dict[observations[obs_idx + 1]]] * matrix_b[obs_idx + 1][to_state]
        HMM.normalize(state_trans_cube)

        new_trans = np.zeros((self.states_size, self.states_size))
        for i in range(self.states_size):
            for j in range(self.states_size):
                for k in range(length - 1):
                    new_trans[i][j] += state_trans_cube[k][i][j]
        HMM.normalize(new_trans)

        new_emit = np.zeros((self.states_size, self.observations_size))
        for i in range(self.states_size):
            for j in range(self.observations_size):
                for k in range(length):
                    if observations[k] == self.observations[j]:
                        new_emit[i][j] += matrix_c[k][i]
        HMM.normalize(new_emit)

        self.addModel(HMM.createModel(matrix_c[0], new_trans, new_emit))


    def printModel(self):
        print 'States:', self.states
        print 'Observations:', self.observations
        for each in self.model_list:
            print '--------Model--------'
            print 'Pi:'
            print each['init']
            print 'Trans:'
            print each['trans']
            print 'Emit:'
            print each['emit']
            print '---------------------'


    @staticmethod
    def extractFlagStr(flag_str):
        flag_dict = {}
        for i in range(len(flag_str)):
            flag_dict[flag_str[i]] = i
        return flag_dict

    @staticmethod
    def laplaceSmoothing(obj, k = 3):
        obj += 3

    @staticmethod
    def normalize(matrix):
        if len(matrix) != 0:
            ele_type = type(matrix[0])
            if ele_type is not list and ele_type is not np.ndarray:
                total = np.sum(matrix)
                if total != 0:
                    for i in range(len(matrix)):
                        matrix[i] /= total
            else:
                for each in matrix:
                    HMM.normalize(each)
        return matrix

class CombinationHMM(HMM):

    def __init__(self, state_segment_list, observation_segment_list):
        HMM.__init__(self, CombinationHMM.generatePermutation(state_segment_list), CombinationHMM.generatePermutation(observation_segment_list))
        self.flag_size = len(state_segment_list)

    def initWithFunction(self, init_func, trans_func, emit_func):
        self.addModel(
            HMM.createModel(
                self.generateInitMatrix(init_func),
                self.generateTransMatrix(trans_func),
                self.generateEmitMatrix(emit_func),
            )
        )

    @staticmethod
    def generatePermutationRec(flag_list, flag_str, flag_segment_list, index):
        if len(flag_segment_list) == index:
            flag_list.append(flag_str)
        else:
            for each in flag_segment_list[index]:
                flag_str += each
                CombinationHMM.generatePermutationRec(flag_list, flag_str, flag_segment_list, index + 1)
                flag_str = flag_str[:-1]

    @staticmethod
    def generatePermutation(flag_segment_list):
        flag_list = []
        flag_str = ''
        CombinationHMM.generatePermutationRec(flag_list, flag_str, flag_segment_list, 0)
        return flag_list

    def generateInitMatrix(self, init_func):
        init_matrix = np.zeros(self.states_size)
        for i in range(self.states_size):
            init_matrix[i] = init_func(self.states[i])
        return HMM.normalize(init_matrix)

    def generateTransMatrix(self, trans_func):
        trans_matrix = np.zeros((self.states_size, self.states_size))
        for i in range(self.states_size):
            for j in range(self.states_size):
                trans_matrix[i, j] = trans_func(self.states[i], self.states[j])
        return HMM.normalize(trans_matrix)

    def generateEmitMatrix(self, emit_func):
        emit_matrix = np.zeros((self.states_size, self.observations_size))
        for i in range(self.states_size):
            for j in range(self.observations_size):
                emit_matrix[i, j] = emit_func(self.states[i], self.observations[j])
        return HMM.normalize(emit_matrix)

    def decode(self, observations):
        decode_str = HMM.decode(self, observations)
        return [decode_str[i:i + self.flag_size] for i in range(0, len(decode_str), self.flag_size)]

if __name__ == '__main__':

    ###############
    # HMM Testing #
    ###############

    hmm = HMM('-o', '._o')

    # Initializing with setting parameters

    # hmm.initWithSetting(
    #     [0.5, 0.5],
    #     [[0.9, 0.1], [0.1, 0.9]],
    #     [[1, 1, 1], [1, 1, 1]],
    # )

    # or with train data.
    hmm.initWithData(
        ['------ooooooooo------'],
        ['...___ooo___ooo___...'],
    )

    # Decode
    print hmm.decode('._._._._ooooo_._ooooo_._._._.')

    # Polish
    hmm.polish('._._._._ooooo_._ooooo_._._._.')
    print hmm.decode('._._._._ooooo_._ooooo_._._._.')

    # Print the mode
    hmm.printModel()

    # Predict
    print hmm.predict('._._._._ooooo_._ooooo_._._._.', 2)

    ##########################
    # CombinationHMM Testing #
    ##########################

    combination_hmm = CombinationHMM(
        ['-o', '12', '12'],
        ['._o', '12', '12'],
    )

    def init_func(state):
        return 1

    def trans_func(state_before, state_after):
        prob = 1
        for i in range(3):
            prob *= (0.9 - 0.1 * i) if state_before[i] == state_after[i] else 0.1 * (i + 1)
        return prob

    def emit_func(state, observation):
        prob = 100 if state[0] == observation[0] else 1
        prob *= 1 - np.sum([math.fabs(int(state[i]) - int(observation[i])) for i in range(1, 3)]) * 0.4
        return prob

    # Initializing with functions above.
    combination_hmm.initWithFunction(init_func, trans_func, emit_func)

    # Print the mode
    combination_hmm.printModel()

    # Decode
    print combination_hmm.decode(['.11', '_22', 'o21', 'o22', '_21', 'o22', 'o11', '_12', '.11'])

    # Polish
    combination_hmm.polish(['.11', '_22', 'o21', 'o22', '_21', 'o22', 'o11', '_12', '.11'])
    print combination_hmm.decode(['.11', '_22', 'o21', 'o22', '_21', 'o22', 'o11', '_12', '.11'])

    # Print the mode
    combination_hmm.printModel()

    # Predict
    print combination_hmm.predict(['.11', '_22', 'o21', 'o22', '_21', 'o22', 'o11', '_12', '.11'], 2)
