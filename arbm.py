#coding:utf-8

import numpy as np
import pickle

class ARBM():
    def initialize(self, visible_size, hidden_size, num_s):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.num_s = num_s
        self.W = np.random.normal(0, 0.1, [visible_size, hidden_size])
        self.b = np.zeros([visible_size, 1], dtype = "float")
        self.c = np.zeros([hidden_size, 1], dtype = "float")
        self.z = np.zeros([visible_size, 1], dtype = "float")

        self.A = np.random.normal(0, 0.1, [visible_size, visible_size, num_s])
        self.B = np.random.normal(0, 0.1, [visible_size, hidden_size, num_s])
        for i in range(num_s):
            vec = np.random.rand(visible_size)
            self.A[:,:,i] = np.diag(vec)


    def save(self, filename):
        params = {}
        params["size"] = [self.visible_size, self.hidden_size, self.num_s]
        params["W"] = self.W
        params["A"] = self.A
        params["B"] = self.B
        params["b"] = self.b
        params["c"] = self.c
        params["z"] = self.z

        with open(filename, "wb") as f:
            pickle.dump(params, f)


    def load(self, filename):
        with open(filename, "rb") as f:
            params = pickle.load(f)

        size_list = params["size"]
        self.visible_size = size_list[0]
        self.hidden_size = size_list[1]
        self.num_s = size_list[2]
        self.W = params["W"]
        self.A = params["A"]
        self.B = params["B"]
        self.b = params["b"]
        self.c = params["c"]
        self.z = params["z"]


    def my_exp(self, array):
        array = np.nan_to_num(array)
        array[np.where(array > 50)] = 50
        array[np.where(array < -50)] = -50
        return np.exp(array)


    def get_hidden_prob(self, visible, s):
        weight = np.dot(self.A[:,:,s], self.W) + self.B[:,:,s]
        lam = self.c + np.dot(weight.T, visible / self.my_exp(self.z))
        prob = 1 / (1 + self.my_exp(-lam))

        return prob


    def get_hidden_sample(self, prob):
        size = prob.shape[0]
        sample = np.zeros([size, 1])

        for i in range(size):
            if np.random.rand() <= prob[i]:
                sample[i] = 1
            else:
                sample[i] = 0

        return sample


    def get_visible_sample(self, hidden, s):
        weight = np.dot(self.A[:,:,s], self.W) + self.B[:,:,s]
        mean = self.b + np.dot(weight, hidden)
        sample = np.random.normal(loc = mean, scale = self.my_exp(self.z) + np.spacing(1), size = [self.visible_size, 1])

        return sample


    def get_visible_value(self, hidden_prob, s):
        weight = np.dot(self.A[:,:,s], self.W) + self.B[:,:,s]
        return self.b + np.dot(weight, hidden_prob)


    def train(self, rate, v_input, s):
        batch_size = v_input.shape[2]
        visible_size = self.visible_size
        hidden_size = self.hidden_size
        update_W = np.zeros([visible_size, hidden_size])
        update_A = np.zeros([visible_size, visible_size])
        update_B = np.zeros([visible_size, hidden_size])
        update_b = np.zeros([visible_size, 1])
        update_c = np.zeros([hidden_size, 1])
        update_z = np.zeros([visible_size, 1])


        weight = np.dot(self.A[:,:,s], self.W) + self.B[:,:,s]
        for i in range(batch_size):
            visible0 = v_input[:,:,i]
            prob0 = self.get_hidden_prob(visible0, s)
            hidden0 = self.get_hidden_sample(prob0)
            visible1 = self.get_visible_sample(hidden0, s)
            prob1 = self.get_hidden_prob(visible1, s)


            for j in range(self.num_s):
                update_W += np.dot(np.dot(self.A[:,:,s].T, visible0), hidden0.T) \
                    - np.dot(np.dot(self.A[:,:,s].T, visible1), prob1.T)

            update_A += np.dot(np.dot(weight, hidden0), visible0.T) \
                - np.dot(np.dot(weight, prob1), visible1.T)

            update_B += np.dot(visible0, hidden0.T) \
                - np.dot(visible1, prob1.T)

            sub = np.nan_to_num(visible0 - self.b)
            sub[np.where(sub > 1e+5)] = 1e+5
            exp_data = np.square(sub) / 2 - visible0 * np.dot(weight, hidden0)

            sub = np.nan_to_num(visible1 - self.b)
            sub[np.where(sub > 1e+5)] = 1e+5
            exp_model = np.square(sub) / 2 - visible1 * np.dot(weight, prob1)
            update_z += exp_data - exp_model
            update_b += visible0 - visible1
            update_c += hidden0 - prob1


        exp = self.my_exp(self.z)
        self.W += rate * update_W / (batch_size * exp)
        self.A[:,:,s] += rate * np.diag(np.diag(update_A)) / (batch_size * exp)
        #self.A[:,:,s] += rate * update_A / batch_size
        self.B[:,:,s] += rate * update_B / batch_size

        self.b += rate * update_b / (batch_size * exp)
        self.c += rate * update_c / batch_size
        self.z += rate * update_z / (batch_size * exp)



    def train_w_o_weight(self, rate, v_input, s):
        batch_size = v_input.shape[2]
        visible_size = self.visible_size
        hidden_size = self.hidden_size
        update_A = np.zeros([visible_size, visible_size])
        update_B = np.zeros([visible_size, hidden_size])

        weight = np.dot(self.A[:,:,s], self.W) + self.B[:,:,s]
        for i in range(batch_size):
            visible0 = v_input[:,:,i]
            prob0 = self.get_hidden_prob(visible0, s)
            hidden0 = self.get_hidden_sample(prob0)
            visible1 = self.get_visible_sample(hidden0, s)
            prob1 = self.get_hidden_prob(visible1, s)

            update_A += (np.dot(np.dot(weight, hidden0), visible0.T) - np.dot(np.dot(weight, prob1), visible1.T))

            update_B += np.dot(visible0, hidden0.T) - np.dot(visible1, prob1.T)


        exp = self.my_exp(self.z)
        self.A[:,:,s] += rate * np.diag(np.diag(update_A)) / (batch_size * exp)
        #self.A[:,:,s] += rate * update_A / batch_size
        self.B[:,:,s] += rate * update_B / batch_size


    def construct(self, visible, input_s, output_s):
        hidden_prob = self.get_hidden_prob(visible, input_s)
        return self.get_visible_value(hidden_prob, output_s)




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    visible_size = 32
    hidden_size = 256
    num_s = 2
    model = ARBM()
    model.initialize(visible_size, hidden_size, num_s)

    #prepare data
    v1 = np.empty((visible_size, 0), "float")
    v2 = np.empty((visible_size, 0), "float")
    size = 50
    for i in range(size):
        v1 = np.hstack([v1, np.random.normal(0, 0.5, [visible_size, 1])])
        v2 = np.hstack([v2, np.random.normal(0, 1, [visible_size, 1])])

    
    #training
    error_list = np.empty(0)
    batch_size = 10
    repeat_num = 100
    rate = 0.0003
    for i in range(repeat_num):
        for j in range(int(np.ceil(size / float(batch_size)))):
            start = j * batch_size
            end = (j + 1) * batch_size
            model.train(rate, v1[:, start:end].reshape(visible_size, 1, batch_size), 0)
            model.train(rate, v2[:, start:end].reshape(visible_size, 1, batch_size), 1)

        #calc error
        err = 0.0
        for k in range(size):
            input_val = v1[:,k].reshape(visible_size, 1)
            err += np.sum(np.square(input_val - model.construct(input_val, 0, 0)))
            input_val = v2[:,k].reshape(visible_size, 1)
            err += np.sum(np.square(input_val - model.construct(input_val, 1, 1)))
        err /= (size * 2)
        error_list = np.hstack([error_list, err])
        print "i = " + str(i) + ": " + str(err)

    #plot error
    plt.plot(error_list)
    plt.xlabel("Training Step")
    plt.ylabel("Error")
    plt.show()
