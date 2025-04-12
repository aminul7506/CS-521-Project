import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import copy
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    budget = 1
    best_result_valid_upto_r20 = 0.0
    best_results_valid = {}
    results_test = {}
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[Valid]")
            results_valid = Procedure.Valid(dataset, Recmodel, epoch, w, world.config['multicore'])
            print("------ Current Valid Results -----")
            print(results_valid)
            if epoch < 1:
                continue
            recall_upto20_valid = (results_valid['recall'][5] + results_valid['recall'][4] +
                                   results_valid['recall'][3] + results_valid['recall'][2])
            if recall_upto20_valid > best_result_valid_upto_r20:
                best_result_valid_upto_r20 = recall_upto20_valid
                best_results_valid = copy.deepcopy(results_valid)
                budget = 5
                print("------ Updating Best Results -----")
                print(best_results_valid)

                results_test = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                print("------ Current Test Results -----")
                print(results_test)
            else:
                budget = budget - 1
                if budget == 0:
                    print("No more training budget")
                    # exit(0)
                    break

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')

    print("------ Best Valid Results -----")
    print(best_results_valid)
    print("------ Final Test Results -----")
    print(results_test)

    # Added by Aminul -- next two blocks
    # if we use pretrained MF for propensity weight, then saving the model
    if world.model_name == 'lgn':
        torch.save(Recmodel.state_dict(), "../data/" + world.dataset + "/model_state_dict_lgn.pth")
    else:
        torch.save(Recmodel.state_dict(), "../data/" + world.dataset + "/model_state_dict.pth")

    # saving the best result
    if world.model_name == 'lgn':
        with open("../data/" + world.dataset + "/results_lgn" + ".txt", "w") as file:
            file.write(str(results_test))
    elif world.model_name == 'mf':
        with open("../data/" + world.dataset + "/results_mf.txt", "w") as file:
                file.write(str(results_test))
    else:
        with open("../../data/" + world.dataset + "/results_unknown.txt", "w") as file:
                file.write(str(results_test))
finally:
    if world.tensorboard:
        w.close()