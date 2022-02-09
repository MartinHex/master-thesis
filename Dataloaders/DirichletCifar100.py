import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

class DirichletCifar100():
    '''
        Use a heirichical Dirichlet sampling to sample natural clients of the CIFAR-100 dataset, as proposed
        by Reddi et al. (https://arxiv.org/pdf/2003.00295.pdf)
    '''
    def __init__(self, number_of_clients, alpha = 0.1, beta = 10):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.map_coarse_to_fine_label = {
            0: [4, 30, 55, 72, 95], # 0
            1: [1, 32, 67, 73, 91], # 1
            2: [54, 62, 70, 82, 92], # 2
            3: [9, 10, 16, 28, 61], # 3
            4 : [0, 51, 53, 57, 83], # 4
            5 : [22, 39, 40, 86, 87], # 5
            6 : [5, 20, 25, 84, 94], # 6
            7 : [6, 7, 14, 18, 24], # 7
            8 : [3, 42, 43, 88, 97], # 8
            9 : [12, 17, 37, 68, 76], # 9
            10 : [23, 33, 49, 60, 71], # 10
            11 : [15, 19, 21, 31, 38], # 11
            12 : [34, 63, 64, 66, 75], # 12
            13 : [26, 45, 77, 79, 99], # 13
            14 : [2, 11, 35, 46, 98], # 14
            15 : [27, 29, 44, 78, 93], # 15
            16 : [36, 50, 65, 74, 80], # 16
            17 : [47, 52, 56, 59, 96], # 17
            18 : [8, 13, 48, 58, 90], # 18
            19 : [41, 69, 81, 85, 89], # 19
        }
        
        self.number_of_clients = number_of_clients
        self.alpha = alpha
        self.beta = beta

        self.trainset = torchvision.datasets.CIFAR100(root= './data', train = True, download = False, transform = self.transform)
        self.testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = False, transform = self.transform)
        
        
    def get_trainset(self):
        class_dict = self._create_class_dict()
        
        number_of_samples = int(len(self.trainset) / self.number_of_clients) # N
        client_sets = []
                                
        available_coarse_label = [x for x in range(20)] # G(r)
        available_fine_label = self.map_coarse_to_fine_label.copy() #G(c)
        
        clients = []

        for client in range(self.number_of_clients):
            print("Currently on client {}/{}".format(client + 1, self.number_of_clients), end = '\r')
            client_samples = []
            theta_r = np.random.dirichlet([self.alpha] * len(available_coarse_label))
            fine_label_distribution = dict()

            for label, values in available_fine_label.items():
                fine_label_distribution[label] = np.random.dirichlet([self.beta] * len(values)) # theta_c

            for i in range(number_of_samples):
                coarse_label_label = np.nonzero(np.random.multinomial(1, theta_r, size = 1))[1][0]
                coarse_label = available_coarse_label[coarse_label_label]

                fine_label_label = np.nonzero(np.random.multinomial(1, fine_label_distribution[coarse_label], size = 1))[1][0]
                fine_label = available_fine_label[coarse_label][fine_label_label]

                sample_index = class_dict[fine_label].pop()

                client_samples.append((
                    self.trainset[sample_index][0],
                    coarse_label,
                    fine_label,
                ))

                # If no samples for a fine label is left remove the fine label
                if (len(class_dict[fine_label]) == 0):
                     # Remove the fine label from the current distribution (theta_c)
                    available_fine_label[coarse_label].remove(fine_label)
                     # Remove the fine label from available fine labels
                    fine_label_distribution[coarse_label] = fine_label_distribution[coarse_label][fine_label_distribution[coarse_label] != fine_label_distribution[coarse_label][fine_label_label]]

                    #renomalize
                    self._renormalize(fine_label_distribution[coarse_label])

                    # If no fine labels for this coarse label is left, remove the coarse label
                    if (len(available_fine_label[coarse_label]) == 0):
                        #Remove coarse label from current distribution
                        theta_r = np.delete(theta_r, coarse_label_label)
                        # Remove coarse label from future available labels
                        available_coarse_label.remove(coarse_label)

                        self._renormalize(theta_r)


            clients.append(client_samples)
        return clients
        
        
    def _create_class_dict(self):
        # Map all the images into a dict with it's targets as key.
        class_dict = dict()
        for i in range(100):
            class_dict[i] = []

        for i in range(len(self.trainset.targets)):
            class_dict[self.trainset.targets[i]].append(i)

        # Make the index list randomly shuffled --> unifrom random distributed
        for key, value in class_dict.items():
            np.random.shuffle(value) 
        return class_dict
    
    def _renormalize(self, theta):
        s = 0
        for l in theta:
            s += l
        map(lambda x: x/s, theta)