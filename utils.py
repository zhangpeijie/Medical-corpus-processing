import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    BOD = get_bod_entity(tag_seq, char_seq)
    SYM = get_sym_entity(tag_seq, char_seq)
    TES= get_tes_entity(tag_seq, char_seq)
    DIS = get_dis_entity(tag_seq, char_seq)
    return BOD, SYM, TES,DIS


def get_bod_entity(tag_seq, char_seq):
    length = len(char_seq)
    BOD = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-bod':
            if 'bod' in locals().keys():
                BOD.append(bod)
                del bod
            bod = char
            if i+1 == length:
                BOD.append(bod)
        if tag == 'I-bod':
            bod += char
            if i+1 == length:
                BOD.append(bod)
        if tag not in ['I-bod', 'B-bod']:
            if 'bod' in locals().keys():
                BOD.append(bod)
                del bod
            continue
    return BOD


def get_sym_entity(tag_seq, char_seq):
    length = len(char_seq)
    SYM = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-sym':
            if 'sym' in locals().keys():
                SYM.append(sym)
                del sym
            sym = char
            if i+1 == length:
                SYM.append(sym)
        if tag == 'I-sym':
            sym += char
            if i+1 == length:
                SYM.append(sym)
        if tag not in ['I-sym', 'B-sym']:
            if 'loc' in locals().keys():
                SYM.append(sym)
                del sym
            continue
    return SYM


def get_dis_entity(tag_seq, char_seq):
    length = len(char_seq)
    DIS = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-DIS':
            if 'dis' in locals().keys():
                DIS.append(dis)
                del dis
            dis = char
            if i+1 == length:
                DIS.append(dis)
        if tag == 'I-ORG':
            dis += char
            if i+1 == length:
                DIS.append(dis)
        if tag not in ['I-dis', 'B-dis']:
            if 'dis' in locals().keys():
                DIS.append(dis)
                del dis
            continue
    return DIS

def get_tes_entity(tag_seq, char_seq):
    length = len(char_seq)
    TES = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-tes':
            if 'tre' in locals().keys():
                TES.append(tes)
                del tes
            tes = char
            if i+1 == length:
                TES.append(tes)
        if tag == 'I-tes':
            tes += char
            if i+1 == length:
                TES.append(tes)
        if tag not in ['I-tes', 'B-tes']:
            if 'tes' in locals().keys():
                TES.append(tes)
                del tes
            continue
    return TES

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
