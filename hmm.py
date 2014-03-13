class HMM:

    def __init__(self, states, observations):
        self.states = states
        self.states_size = len(states)
        self.states_dict = HMM.extractFlagStr(states)
        self.observations = observations
        self.observations_size = len(observations)
        self.observations_dict = HMM.extractFlagStr(observations)
        self.model_list = []
        self.model_list_index = -1

    @staticmethod
    def createModel(init, trans, emit):
        model = {}
        model['init'] = init
        model['trans'] = trans
        model['emit'] = emit
        return model

    def initWithSetting(self, init, trans, emit):
        self.model_list.append(HMM.createModel(init, trans, emit))
        self.model_list_index = 0

    def initWithData(self, states_list, observations_list):
        model = HMM.createModel(HMM.createList(self.states_size),
            HMM.createMatrix(self.states_size, self.states_size),
            HMM.createMatrix(self.states_size, self.observations_size))

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

        model['init'] = HMM.laplaceSmoothing(model['init'])
        model['trans'] = HMM.laplaceSmoothing(model['trans'])
        model['emit'] = HMM.laplaceSmoothing(model['emit'])

        HMM.normalize(model['init'])
        HMM.normalizeMatrix(model['trans'])
        HMM.normalizeMatrix(model['emit'])

        self.model_list.append(model)
        self.model_list_index = 0

    def getCurrentModel(self):
        return self.model_list[self.model_list_index]

    def calculateA(self, observations):
        model = self.getCurrentModel()
        length = len(observations)
        matrix_a = HMM.createMatrix(length, self.states_size)

        for state in range(self.states_size):
            matrix_a[0][state] = model['init'][state] * model['emit'][state][self.observations_dict[observations[0]]]
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
        matrix_b = HMM.createMatrix(length, self.states_size)

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
        matrix_c = HMM.createMatrix(length, self.states_size)
        for obs_idx in range(length):
            for state_idx in range(self.states_size):
                matrix_c[obs_idx][state_idx] = matrix_a[obs_idx][state_idx] * matrix_b[obs_idx][state_idx]
        HMM.normalizeMatrix(matrix_c)
        return matrix_a, matrix_b, matrix_c

    def filter(self, observations):
        pass

    def smooth(self, observations):
        pass

    def predict(self, observations, step = 1):
        model = self.getCurrentModel()
        prob = self.calculateA(observations)[-1]
        for s in range(step):
            prob_next = HMM.createList(self.states_size)
            for i in range(self.states_size):
                for j in range(self.states_size):
                    prob_next[i] += prob[j] * model['trans'][j][i]
            HMM.normalize(prob_next)
            prob = prob_next
        return self.states[HMM.getMaxIndex(prob)]

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
        return state_decode[HMM.getMaxIndex(state_prob)]

    def polish(self, observations):
        model = self.getCurrentModel()
        matrix_a, matrix_b, matrix_c = self.calculateObsProbMatrixs(observations)
        length = len(observations)
        state_trans_cube = HMM.createCube(length - 1, self.states_size, self.states_size)
        for obs_idx in range(length - 1):
            for from_state in range(self.states_size):
                for to_state in range(self.states_size):
                    state_trans_cube[obs_idx][from_state][to_state] = matrix_a[obs_idx][from_state] * model['trans'][from_state][to_state] * model['emit'][to_state][self.observations_dict[observations[obs_idx + 1]]] * matrix_b[obs_idx + 1][to_state]
        HMM.normalizeCube(state_trans_cube)

        new_trans = HMM.createMatrix(self.states_size, self.states_size)
        for i in range(self.states_size):
            for j in range(self.states_size):
                for k in range(length - 1):
                    new_trans[i][j] += state_trans_cube[k][i][j]
        HMM.normalizeMatrix(new_trans)

        new_emit = HMM.createMatrix(self.states_size, self.observations_size)
        for i in range(self.states_size):
            for j in range(self.observations_size):
                for k in range(length):
                    if observations[k] == self.observations[j]:
                        new_emit[i][j] += matrix_c[k][i]
        HMM.normalizeMatrix(new_emit)

        self.model_list.append(HMM.createModel(matrix_c[0], new_trans, new_emit))
        self.model_list_index += 1

    def printModel(self):
        print 'States:', self.states
        print 'Observations:', self.observations
        for each in self.model_list:
            print '--------Model--------'
            print 'Pi:', each['init']
            print 'Trans:'
            HMM.printMatrix(each['trans'])
            print 'Emit:'
            HMM.printMatrix(each['emit'])
            print '---------------------'

    @staticmethod
    def extractFlagStr(flag_str):
        flag_dict = {}
        for i in range(len(flag_str)):
            flag_dict[flag_str[i]] = i
        return flag_dict

    @staticmethod
    def createList(n):
        vector = []
        for i in range(n):
            vector.append(0)
        return vector

    @staticmethod
    def createMatrix(n, m):
        matrix = []
        for i in range(n):
            matrix.append(HMM.createList(m))
        return matrix

    @staticmethod
    def createCube(n, m, l):
        cube = []
        for i in range(n):
            cube.append(HMM.createMatrix(m, l))
        return cube

    @staticmethod
    def getMaxIndex(vector):
        index = 0
        for i in range(len(vector)):
            if vector[i] > vector[index]:
                index = i
        return index

    @staticmethod
    def laplaceSmoothing(obj, k = 3):
        obj_type = type(obj);
        if int == obj_type or float == obj_type:
            return obj + k
        elif list == obj_type:
            for i in range(len(obj)):
                obj[i] = HMM.laplaceSmoothing(obj[i], k)
            return obj

    @staticmethod
    def normalize(vector):
        total = 0.0
        for each in vector:
            total += each
        length = len(vector)
        if total != 0:
            for i in range(length):
                vector[i] /= total
        # else:
        #     for i in range(length):
        #         vector[i] = 1.0 / length

    @staticmethod
    def normalizeMatrix(matrix):
        for each in matrix:
            HMM.normalize(each)

    @staticmethod
    def normalizeCube(cube):
        for each in cube:
            HMM.normalizeMatrix(each)

    @staticmethod
    def printMatrix(matrix, indent = 1):
        print ' ' * ((indent - 1) * 4) + '['
        for each in matrix:
            print ' ' * (4 * indent) + str(each)
        print ' ' * ((indent - 1) * 4) + ']'

    @staticmethod
    def printCube(cube):
        print '['
        for each in cube:
            HMM.printMatrix(each, 2)
        print ']'

if __name__ == '__main__':
    hmm = HMM('-o', '._o')

    # initializing with setting parameters
    # hmm.initWithSetting([1, 0, 0], [[0, 0.5, 0.5], [0, 0.9, 0.1], [0, 0, 1]], [[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
    # or with train data
    hmm.initWithData(['------ooooooooo------'], ['...___ooo___ooo___...'])

    # decode
    print hmm.decode('._._._._ooooo_._ooooo_._._._.')

    # polish
    hmm.polish('._._._._ooooo_._ooooo_._._._.')
    print hmm.decode('._._._._ooooo_._ooooo_._._._.')

    # print the mode
    hmm.printModel()

    # predict
    # print hmm.predict('._._._._ooooo_._ooooo_._._._.', 2)
