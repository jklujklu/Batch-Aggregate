import hashlib
import json
import sys
import time
from functools import reduce

import numpy as np
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair

debug = False


class ShortSig(object):

    @staticmethod
    def pubkey(group):
        # init
        g1, g2 = group.random(G1), group.random(G2)
        h = group.random(G1)
        s1, s2 = group.random(ZR), group.random(ZR)
        u, v = h * ~s1, h * ~s2
        gamma = group.random(ZR)
        _lambda = group.random(ZR)
        w = g2 * gamma
        gpk = {'g1': g1, 'g2': g2, 'h': h, 'u': u, 'v': v, 'w': w, 'lambda': _lambda}
        gmsk = {'s1': s1, 's2': s2}

        return gpk, gmsk, gamma

    @staticmethod
    def keygen(group, gpk, gamma):
        # key gen
        x = group.random(ZR)
        A = gpk['g1'] * ~(gamma + x)
        return A, x

    @staticmethod
    def sign(group, gpk, gsk, M):
        alpha, beta = group.random(), group.random()
        A, x = gsk[0], gsk[1]
        T1 = gpk['u'] * alpha
        T2 = gpk['v'] * beta
        T3 = A + (gpk['h'] * (alpha + beta))
        delta1 = x * alpha
        delta2 = x * beta
        r = [group.random() for i in range(5)]
        R1 = gpk['u'] * r[0]
        R2 = gpk['v'] * r[1]
        R3 = (pair(T3, gpk['g2']) ** r[2]) * (pair(gpk['h'], gpk['w'] * (-r[0] - r[1]) + gpk['g2'] * (-r[3] - r[4])))
        R4 = (T1 * r[2]) - (gpk['u'] * r[3])
        R5 = (T2 * r[2]) - (gpk['v'] * r[4])
        timestamp = int(time.time())
        c = group.hash((M, T1, T2, T3, R1, R2, R3, R4, R5, timestamp), ZR)
        s1, s2 = r[0] + c * alpha, r[1] + c * beta
        s3, s4 = r[2] + c * x, r[3] + c * delta1
        s5 = r[4] + c * delta2
        return {'T1': T1, 'T2': T2, 'T3': T3, 'c': c, 's_alpha': s1, 's_beta': s2, 's_x': s3, 's_delta1': s4,
                's_delta2': s5, 'R3': R3, }, timestamp

    @staticmethod
    def verify(group, gpk, msg):
        validSignature = False
        sigma, msg, timestamp = msg
        c, t1, t2, t3 = sigma['c'], sigma['T1'], sigma['T2'], sigma['T3']
        s_alpha, s_beta = sigma['s_alpha'], sigma['s_beta']
        s_x, s_delta1, s_delta2 = sigma['s_x'], sigma['s_delta1'], sigma['s_delta2']

        R1_ = (gpk['u'] * s_alpha) + (t1 * -c)
        R2_ = (gpk['v'] * s_beta) + (t2 * -c)
        R3_ = (pair(t3 * s_x, gpk['g2'])) * (pair(c * t3, gpk['w'])) * \
              (pair(gpk['h'], gpk['w']) ** (-s_alpha - s_beta)) * \
              (pair(gpk['h'], gpk['g2']) ** (-s_delta1 - s_delta2)) * \
              (pair(gpk['g1'], gpk['g2']) ** -c)
        R4_ = (t1 * s_x) - (gpk['u'] * s_delta1)
        R5_ = (t2 * s_x) - (gpk['v'] * s_delta2)

        c_prime = group.hash((msg, t1, t2, t3, R1_, R2_, R3_, R4_, R5_, timestamp), ZR)
        if c == c_prime:
            if debug: print("c => '%s'" % c)
            if debug: print("Valid Group Signature for message: '%s'" % msg)
            validSignature = True
        else:
            if debug: print("Not a valid signature for message!!!")

        return validSignature

    @staticmethod
    def batch_verify(group, gpk, msgs):
        sigmas = []
        for _ in msgs:
            sign, msg, timestamp = _
            sigmas.append(sign)
            sign['delta'] = group.random(ZR)
            r1_ = gpk['u'] * sign['s_alpha'] + sign['T1'] * -sign['c']
            r2_ = gpk['v'] * sign['s_beta'] + sign['T2'] * -sign['c']
            r4_ = sign['T1'] * sign['s_x'] - gpk['u'] * sign['s_delta1']
            r5_ = sign['T2'] * sign['s_x'] - gpk['v'] * sign['s_delta2']
            c_ = group.hash((msg, sign['T1'], sign['T2'], sign['T3'], r1_, r2_, sign['R3'], r4_, r5_, timestamp),
                            ZR)
            if c_ != sign['c']:
                print('False')
                return False

        r3_left = []
        r3_right = []
        r3 = 1
        for sign in sigmas:
            r3_left.append((sign['T3'] * sign['s_x'] - (sign['s_delta1'] + sign['s_delta2']) * gpk['h'] - gpk[
                'g1'] * sign['c']) * sign['delta'])
            r3_right.append((sign['T3'] * sign['c'] - (sign['s_alpha'] + sign['s_beta']) * gpk['h']) * sign['delta'])
            r3 *= (sign['R3'] * sign['delta'])
        r3_ = (pair(reduce(lambda x, y: x + y, r3_left), gpk['g2'])) * (
            pair(reduce(lambda x, y: x + y, r3_right), gpk['w']))
        # r3 = reduce(lambda x, y: (x['R3'] * x['delta']) * (y['R3'] * y['delta']), sigmas)

        if r3_ == r3:
            return True
        else:
            return False


if __name__ == '__main__':
    group = PairingGroup('SS80')

    # group = PairingGroup('SS512')
    g1 = group.random(G1)
    g2 = group.random(G1)

    tic = time.time()
    gt = pair(g1, g2)
    for i in range(1000):
        gt ** group.random(ZR)
    print(round((time.time() - tic) * 1000, 2))
    # print(g1, g2, gt)
    # g2 = group.random(G2)
    # print(g1)
    # print(g2)
    # print(g1 / g2)
    # # print(g1 * 0.1211)
    # # b = group.random()
    # # print(pair(g1 ** a,g2 ** b))
    # # print(pair(g1, g2) ** (a * b))
    #
    # n = 100  # how manu users are in the group
    # user = 1  # which user's key we will sign a message with
    # shortSig = ShortSig(group)
    #
    # (global_public_key, global_master_secret_key, user_secret_keys) = shortSig.keygen(n)
    #
    # # times = []
    # # for i in range(9):
    # #     start = time.time()
    # #     msg = get_hash('./j1.json')
    # #     print(msg)
    # #     signature = shortSig.sign(global_public_key, user_secret_keys[user], msg)
    # #     times.append(round((time.time() - start) * 1000, 3))
    # # print(times)
    # msgs = []
    # signs = []
    # for i in range(n):
    #     eng_msg = shortSig.encrypt(np.array([1, 2, 3]), user_secret_keys[i])
    #     signature = shortSig.sign(global_public_key, user_secret_keys[i], eng_msg['e'])
    #     msgs.append(eng_msg)
    #     signs.append(signature)
    #     # msg1_sign = shortSig.verify(global_public_key, enc_msg, signature, user_secret_keys[user])
    # shortSig.batch_verify(global_public_key, msgs, signs, user_secret_keys)
    # #
    # # msg2 = np.array([1, 2, 3])
    # # eng_msg = shortSig.encrypt(msg2, user_secret_keys[user])
    # # signature2 = shortSig.sign(global_public_key, user_secret_keys[user], eng_msg['e'])
    # # shortSig.verify(global_public_key, eng_msg, signature2, user_secret_keys[user])
