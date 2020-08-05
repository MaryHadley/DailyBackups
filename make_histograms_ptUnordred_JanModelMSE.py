import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import uproot
from array import array
import ROOT
from ROOT import TLorentzVector
from collections import OrderedDict
import math
import sys
from my_classes_LF_ptUnordered import DataGenerator
import pickle
import json
import keras.backend as K
import copy
from datetime import datetime, date
import os

today = date.today()
dateToInclude = today.strftime("%b-%d-%Y")
print (dateToInclude)

now = datetime.now()
#print (now)
my_current_time = now.strftime("%H:%M:%S")

##constants###
tauMass = 1.777

tau_feature_names = [
    b'pi_minus1_pt',
    b'pi_minus1_eta',
    b'pi_minus1_phi',
    b'pi_minus2_pt',
    b'pi_minus2_eta',
    b'pi_minus2_phi',
    b'pi_minus3_pt',
    b'pi_minus3_eta',
    b'pi_minus3_phi',
]
tau_label_names = [
    b'neutrino_pt',
    b'neutrino_eta',
    b'neutrino_phi',
]

antitau_feature_names = [
    b'pi_plus1_pt',
    b'pi_plus1_eta',
    b'pi_plus1_phi',
    b'pi_plus2_pt',
    b'pi_plus2_eta',
    b'pi_plus2_phi',
    b'pi_plus3_pt',
    b'pi_plus3_eta',
    b'pi_plus3_phi',
]
antitau_label_names = [
    b'antineutrino_pt',
    b'antineutrino_eta',
    b'antineutrino_phi',
]




with open('partition.txt') as json_file:
    partition = json.load(json_file)
    

params = {'dim': 12,
          'batch_size': 256,
          'num_of_labels': 4,
          'shuffle': False}



training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

big_model = tf.keras.models.load_model('/uscms_data/d3/mhadley/runJanModel/consolidateStuffToUse_WeekOf3August2020/Jan_Models/4August2020Weights_JanModel_MSE_BS1024-weights-improvement-52-0.2678.hdf5') 
pred = big_model.predict_generator(validation_generator, steps = 100) #Batch size 256, 25600 total validation examples


file=uproot.open("fout_particleGun.root")["tree"] 
print("GOT TREE FROM fout_particleGun.root")
tau_features = []
tau_labels = []
antitau_features = []
antitau_labels = []

for name in tau_feature_names:
    if b'_phi' in name:
        tau_features.append(
            np.sin(file.array(name))
        )
        tau_features.append(
            np.cos(file.array(name))
        )
    else:
        tau_features.append(file.array(name))

for name in tau_label_names:
    if b'_phi' in name:
        tau_labels.append(
            np.sin(file.array(name))
        )
        tau_labels.append(
            np.cos(file.array(name))
        )
    else:
        tau_labels.append(file.array(name))

for name in antitau_feature_names:
    if b'_phi' in name:
        antitau_features.append(
            np.sin(file.array(name))
        )
        antitau_features.append(
            np.cos(file.array(name))
        )
    else:
        antitau_features.append(file.array(name))

for name in antitau_label_names:
    if b'_phi' in name:
        antitau_labels.append(
            np.sin(file.array(name))
        )
        antitau_labels.append(
            np.cos(file.array(name))
        )
    else:
        antitau_labels.append(file.array(name))



tau_features_test = np.transpose(np.array(tau_features))
tau_labels_test = np.transpose(np.array(tau_labels))

antitau_features_test = np.transpose(np.array(antitau_features))
antitau_labels_test = np.transpose(np.array(antitau_labels))

print("tau_features_test.shape", tau_features_test.shape)
print("tau_labels_test.shape", tau_labels_test.shape)
print("antitau_features_test.shape", antitau_features_test.shape)
print("antitau_labels_test.shape", antitau_labels_test.shape)

tau_labels_test[:,0] *= (1/tauMass) #Now the labels are what we want, which is local neu (anti neu) pt normalized by tau mass
antitau_labels_test[:,0] *= (1/tauMass)



#print("tau_labels_test", tau_labels_test)
print("Before getting only the first 12800 events")
print("first row of tau feautures test :", tau_features_test[0,:])
print("first row of tau labels test :", tau_labels_test[0,:])
print('tau_labels_test.shape', tau_labels_test.shape)
print('antitau_labels_test.shape', antitau_labels_test.shape)
print('tau_features_test.shape', tau_features_test.shape)
print("antitau_features_test.shape", antitau_features_test.shape)
print('##################')
#play games to get numbering right for when we make the plots

print("After getting only the first 12800 events")
tau_labels_test = tau_labels_test[:12800, :]
antitau_labels_test = antitau_labels_test[:12800, :]
tau_features_test = tau_features_test[:12800, :]
antitau_features_test = antitau_features_test[:12800, :]
print('tau_labels_test.shape', tau_labels_test.shape)
print('antitau_labels_test.shape', antitau_labels_test.shape)
print('tau_features_test.shape', tau_features_test.shape)
print('antitau_features_test.shape', antitau_features_test.shape)

print("first row of tau feautures test:", tau_features_test[0,:])
print("first row of antitau_features_test:", antitau_features_test[0,:])
print("first row of tau labels test", tau_labels_test[0,:])
print("first row of antitau labels test", antitau_labels_test[0,:])

def arr_normalize(arr):
    arr = np.where(arr > 1, 1, arr)
    arr = np.where(arr < -1, -1, arr)
    return arr


def arr_get_angle(sin_value, cos_value):
    sin_value = arr_normalize(sin_value)
    cos_value = arr_normalize(cos_value)
    return np.where(
        sin_value > 0,
        np.where(
            cos_value > 0,
            (np.arcsin(sin_value) + np.arccos(cos_value)) / 2,
            ((np.pi - np.arcsin(sin_value)) + np.arccos(cos_value)) / 2
        ),
        np.where(
            cos_value > 0,
            (np.arcsin(sin_value) - np.arccos(cos_value)) / 2,
            ((- np.arccos(cos_value)) - (np.pi + np.arcsin(sin_value))) / 2
        )
    )

split_pred = np.split(pred,2)
print("split_pred is",split_pred)
print("len(split_pred) is:", len(split_pred))

pred = split_pred[0]
anti_pred = split_pred[1]

print("pred.shape", pred.shape)
print("anti_pred.shape", anti_pred.shape)

def arr_normalize(arr):
    arr = np.where(arr > 1, 1, arr)
    arr = np.where(arr < -1, -1, arr)
    return arr


def arr_get_angle(sin_value, cos_value):
    sin_value = arr_normalize(sin_value)
    cos_value = arr_normalize(cos_value)
    return np.where(
        sin_value > 0,
        np.where(
            cos_value > 0,
            (np.arcsin(sin_value) + np.arccos(cos_value)) / 2,
            ((np.pi - np.arcsin(sin_value)) + np.arccos(cos_value)) / 2
        ),
        np.where(
            cos_value > 0,
            (np.arcsin(sin_value) - np.arccos(cos_value)) / 2,
            ((- np.arccos(cos_value)) - (np.pi + np.arcsin(sin_value))) / 2
        )
    )


pred[:, 2] = arr_get_angle(pred[:, 2], pred[:, 3])
pred = pred[:, 0: 3]

print("pred.shape after conversion to phi", pred.shape)

anti_pred[:, 2] = arr_get_angle(anti_pred[:, 2], anti_pred[:, 3])
anti_pred = anti_pred[:, 0: 3]
print("anti_pred.shape after conversion to phi", anti_pred.shape)

tau_features_test[:, 2] = arr_get_angle(tau_features_test[:, 2], tau_features_test[:, 3])
tau_features_test[:, 6] = arr_get_angle(tau_features_test[:, 6], tau_features_test[:, 7])
tau_features_test[:, 10] = arr_get_angle(tau_features_test[:, 10], tau_features_test[:, 11])
tau_features_test = tau_features_test[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]
print("tau_features_test.shape after conversion to phi", tau_features_test.shape)

tau_labels_test[:, 2] = arr_get_angle(tau_labels_test[:, 2], tau_labels_test[:, 3])
tau_labels_test = tau_labels_test[:, 0: 3]
print("tau_labels_test.shape after conversion to phi", tau_labels_test.shape)

antitau_features_test[:, 2] = arr_get_angle(antitau_features_test[:, 2], antitau_features_test[:, 3])
antitau_features_test[:, 6] = arr_get_angle(antitau_features_test[:, 6], antitau_features_test[:, 7])
antitau_features_test[:, 10] = arr_get_angle(antitau_features_test[:, 10], antitau_features_test[:, 11])
antitau_features_test = antitau_features_test[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]

print("antitau_features_test.shape after conversion to phi", antitau_features_test.shape)

antitau_labels_test[:, 2] = arr_get_angle(antitau_labels_test[:, 2], antitau_labels_test[:, 3])
antitau_labels_test = antitau_labels_test[:, 0: 3]
print("antitau_labels_test.shape after conversion to phi", antitau_labels_test.shape)

big_labels_test = np.concatenate((tau_labels_test, antitau_labels_test)) #You need the double parentheses here and I always forget
big_pred = np.concatenate((pred, anti_pred))
print("big_labels_test.shape", big_labels_test.shape)
print("big_pred.shape is:" ,big_pred.shape)

theDir = dateToInclude + "_" + my_current_time + "_" + "ptUnorderedJanModel_MSE"
if not os.path.exists(dateToInclude + "_" + my_current_time + "_" + "ptUnorderedJanModel_MSE"):
    os.makedirs(dateToInclude + "_" + my_current_time + "_" + "ptUnorderedJanModel_MSE")

# #overall nu plots
plt.plot(big_labels_test[:, 0], big_pred[:, 0], 'ro')
#x=big_labels_test[:, 0]
#y=big_pred[:, 0]
fig1 = plt.gcf()
#xmin, xmax = 0, 1.4
#ymin, ymax = 0, 1.4
#axes = plt.gca()
#axes.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x))) #https://stackoverflow.com/questions/3777861/setting-y-axis-limit-in-matplotlib
#best_fit =  np.poly1d(np.polyfit(x, y, 1))
#print("best_fit.c", best_fit.c) #https://stackoverflow.com/questions/9990789/how-to-force-zero-interception-in-linear-regression
#x = x[:,np.newaxis]
#a, _, _, _ = np.linalg.lstsq(x, y)

#plt.plot(x, y, 'bo')
#plt.plot(x, a*x, 'r-')
#print("slope for pt/tauMass is:", a)
fig1.savefig(theDir +'/big_nu_pt.png') # the variable here is local pT/tau mass
plt.clf()

plt.plot(big_labels_test[:, 1], big_pred[:, 1], 'ro')
fig1 = plt.gcf()
#x=big_labels_test[:, 1]
#y=pred[:, 1]
# xmin, xmax = 0, np.pi
# ymin, ymax =0, 0.8
# x = x[:,np.newaxis]
# a, _, _, _ = np.linalg.lstsq(x, y)
# plt.plot(x, y, 'bo')
# #plt.plot(x, a*x, 'r-')
# print("slope for theta is:", a)
# axes = plt.gca()
# axes.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
fig1.savefig(theDir + '/big_nu_eta.png')
plt.clf()
fig1.savefig(theDir + '/big_nu_eta.png')
plt.clf()


plt.plot(big_labels_test[:, 2], big_pred[:, 2], 'ro')
fig1 = plt.gcf()
plt.clf()
fig1.savefig(theDir + '/big_nu_phi.png')
plt.clf()

max_error = 0
total_error = 0
for x in range(big_labels_test.shape[0]): #get number of rows with shape[0]
    error = abs(big_labels_test[x][0] - big_pred[x][0])
    max_error = max(error, max_error)
    total_error += error

max_error =max_error
mean_error = total_error/big_labels_test.shape[0]

print('max_error for nu pt norm by tau mass is:', max_error)
print("mean_error for nu pt norm by tau mass is:", mean_error)

out_file = open(theDir + '/Errors.txt', 'w')
out_file.write('max_error for nu pt norm by tau mass is %s: \n' %str(max_error))
out_file.write('mean_error for nu pt norm by tau mass is %s: \n' %str(mean_error))
out_file.close()


def normalized_error(x,y):
    diff = abs(x-y)
    if diff <= np.pi:
        return diff
    else:
        return  normalized_error(2*np.pi,diff)

max_error =0
total_error = 0

for x in range(big_labels_test.shape[0]):
#    error = normalized_error(big_labels_test[x][1], big_pred[x][1])
#    max_error = max(error, max_error)
#    total_error += error
    error = abs(big_labels_test[x][1] - big_pred[x][1])
    max_error = max(error, max_error)
    total_error += error
# 
max_error =max_error
mean_error = total_error/big_labels_test.shape[0]

out_file = open(theDir + '/Errors.txt', 'a')
out_file.write('max_error for nu eta %s: \n' %str(max_error))
out_file.write('mean_error for nu eta is %s \n:' %str(mean_error))
out_file.close()

print("max_error for nu eta is:", max_error)
print("mean_error for nu eta is:", mean_error)


max_error = 0
total_error = 0

for x in range(big_labels_test.shape[0]): #get number of rows with shape[0]
#    error = abs(big_labels_test[x][2] - big_pred[x][2])
#    max_error = max(error, max_error)
 #   total_error += error
    error = normalized_error(big_labels_test[x][2], big_pred[x][2])
    max_error = max(error, max_error)
    total_error += error

max_error =max_error
mean_error = total_error/big_labels_test.shape[0]
print('max_error for nu phi is:', max_error)
print('mean_error for nu phi is:', mean_error)

out_file = open(theDir + '/Errors.txt', 'a')
out_file.write('max_error for nu phi %s: \n' %str(max_error))
out_file.write('mean_error for nu phi is %s \n:' %str(mean_error))
out_file.close()

branches = [
    'tau_pt',
    'tau_eta',
    'tau_phi',
    'tau_mass',
    'tau_pt_no_neutrino',
    'tau_eta_no_neutrino',
    'tau_phi_no_neutrino',
    'tau_mass_no_neutrino',
    'antitau_pt',
    'antitau_eta',
    'antitau_phi',
    'antitau_mass',
    'antitau_pt_no_neutrino',
    'antitau_eta_no_neutrino',
    'antitau_phi_no_neutrino',
    'antitau_mass_no_neutrino',
#    'upsilon_pt',
#    'upsilon_eta',
#    'upsilon_phi',
#    'upsilon_mass',
#    'upsilon_pt_no_neutrino',
#    'upsilon_eta_no_neutrino',
#    'upsilon_phi_no_neutrino',
#    'upsilon_mass_no_neutrino',
]
file_out = ROOT.TFile(theDir +'/3_prong_no_neutral_graphs.root', 'recreate')
file_out.cd()

tofill = OrderedDict(zip(branches, [-99.] * len(branches)))

masses = ROOT.TNtuple('tree', 'tree', ':'.join(branches))

# # 
for event in range(pred.shape[0]):
    tau_lorentz_no_neutrino = TLorentzVector()

    for index in range(0, tau_features_test.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
            tau_features_test[event][index],
            tau_features_test[event][index + 1],
            tau_features_test[event][index + 2],
            0.139
        )
        tau_lorentz_no_neutrino += lorentz

    tofill['tau_pt_no_neutrino'] = tau_lorentz_no_neutrino.Pt()
    tofill['tau_eta_no_neutrino'] = tau_lorentz_no_neutrino.Eta()
    tofill['tau_phi_no_neutrino'] = tau_lorentz_no_neutrino.Phi()
    tofill['tau_mass_no_neutrino'] = tau_lorentz_no_neutrino.M()

    tau_lorentz = TLorentzVector()
    tau_lorentz.SetPtEtaPhiM(
        tau_lorentz_no_neutrino.Pt(),
        tau_lorentz_no_neutrino.Eta(),
        tau_lorentz_no_neutrino.Phi(),
        tau_lorentz_no_neutrino.M(),
    )

    for index in range(0, pred.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
            pred[event][index],
            pred[event][index + 1],
            pred[event][index + 2],
            0
        )
        tau_lorentz += lorentz

    tofill['tau_pt'] = tau_lorentz.Pt()
    tofill['tau_eta'] = tau_lorentz.Eta()
    tofill['tau_phi'] = tau_lorentz.Phi()
    tofill['tau_mass'] = tau_lorentz.M()

    antitau_lorentz_no_neutrino = TLorentzVector()

    for index in range(0, tau_features_test.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
            antitau_features_test[event][index],
            antitau_features_test[event][index + 1],
            antitau_features_test[event][index + 2],
            0.139
        )
        antitau_lorentz_no_neutrino += lorentz

    tofill['antitau_pt_no_neutrino'] = antitau_lorentz_no_neutrino.Pt()
    tofill['antitau_eta_no_neutrino'] = antitau_lorentz_no_neutrino.Eta()
    tofill['antitau_phi_no_neutrino'] = antitau_lorentz_no_neutrino.Phi()
    tofill['antitau_mass_no_neutrino'] = antitau_lorentz_no_neutrino.M()

    antitau_lorentz = TLorentzVector()
    antitau_lorentz.SetPtEtaPhiM(
        antitau_lorentz_no_neutrino.Pt(),
        antitau_lorentz_no_neutrino.Eta(),
        antitau_lorentz_no_neutrino.Phi(),
        antitau_lorentz_no_neutrino.M(),
    )

    for index in range(0, anti_pred.shape[1], 3):
        lorentz = TLorentzVector()
        lorentz.SetPtEtaPhiM(
            anti_pred[event][index],
            anti_pred[event][index + 1],
            anti_pred[event][index + 2],
            0
        )
        antitau_lorentz += lorentz

    tofill['antitau_pt'] = antitau_lorentz.Pt()
    tofill['antitau_eta'] = antitau_lorentz.Eta()
    tofill['antitau_phi'] = antitau_lorentz.Phi()
    tofill['antitau_mass'] = antitau_lorentz.M()

#    upsilon_lorentz = tau_lorentz + antitau_lorentz
#   upsilon_lorentz_no_neutrino = tau_lorentz_no_neutrino + antitau_lorentz_no_neutrino

#    tofill['upsilon_pt_no_neutrino'] = upsilon_lorentz_no_neutrino.Pt()
#    tofill['upsilon_eta_no_neutrino'] = upsilon_lorentz_no_neutrino.Eta()
#    tofill['upsilon_phi_no_neutrino'] = upsilon_lorentz_no_neutrino.Phi()
#    tofill['upsilon_mass_no_neutrino'] = upsilon_lorentz_no_neutrino.M()
#    tofill['upsilon_pt'] = upsilon_lorentz.Pt()
#    tofill['upsilon_eta'] = upsilon_lorentz.Eta()
#    tofill['upsilon_mass'] = upsilon_lorentz.M()

    masses.Fill(array('f', tofill.values()))

file_out.cd()
masses.Write()
file_out.Close()
