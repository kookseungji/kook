import neural_net as nn
import redo



#regularization = [1,.1,.01,.001,.0001]
regularization = [.001]
learning_rates = [1]
hidden_layer_sizes = [500]
learning_rate_decays = [.9]
batch_sizes = [250]
num_models_to_train = len(regularization)*len(learning_rates)*len(hidden_layer_sizes)*len(batch_sizes)

num_iters = 5000

if __name__ == '__main__':
    print("Loading and preprocessing data.")
    Xtr, Ytr, Xval, Yval, Xte, Yte = redo.load_and_process()
    max_accuracies = []
    max_accuracy_setups = []
    trained, first  = 0, True
    print('Beginning hyper-parameteriziation')
    for reg in regularization:
        for learning_rate in learning_rates:
            for hidden_layer_size in hidden_layer_sizes:
                for learning_rate_decay in learning_rate_decays:
                    for batch_size in batch_sizes:
                        param_dict = {}
                        param_dict['reg'] = reg
                        param_dict['learning_rate'] = learning_rate
                        param_dict['hidden_layer_size'] = hidden_layer_size
                        param_dict['learning_rate_decay'] = learning_rate_decay
                        param_dict['batch_size'] = batch_size
                        param_dict['num_iters'] = 5000
                        print('Training: '+str(param_dict))
                        results = redo.build_and_train_nn(Xtr, Ytr, Xval, Yval,verbose = True, reg = reg, learning_rate = learning_rate, hidden_layer_size = hidden_layer_size, learning_rate_decay = learning_rate_decay, num_iters = num_iters)
                        trained += 1
                        print('Progress: '+str(float(trained)/num_models_to_train))
                        with open('hyperlog.csv','a+') as csv:
                        	if first == True:
                                	first = False
                                        csv.write('last_accuracy_val, last_training_accuracy_val, loss, num_iters, batch_size, learning_rate_decay, hidden_layer_size, learning_rate, reg,time \n')
                                csv.write(str(results['val_acc_history'][-1])+', ')
				csv.write(str(results['train_acc_history'][-1])+', ')
				csv.write(str(results['loss_history'][-1])+', ')
                   		csv.write(str(param_dict['num_iters'])+', ')
                                csv.write(str(param_dict['batch_size'])+', ')
                                csv.write(str(param_dict['learning_rate_decay'])+', ')
                                csv.write(str(param_dict['hidden_layer_size'])+', ')
                                csv.write(str(param_dict['learning_rate'])+', ')
                                csv.write(str(param_dict['reg'])+', ')
                                csv.write(str(results['time'])+'\n')
                                


                                                               

