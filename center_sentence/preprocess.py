# coding=utf-8
import sys
from imp import reload
reload(sys)
# sys.setdefaultencoding('utf-8')
import difflib

import re

def str_similarity(str1,str2,delta):
    tag = False
    seq = difflib.SequenceMatcher(None,str1,str2 )
    if seq.ratio() >= delta:
        tag = True
    return tag

def cnstr2num(cnstr):
    switch={u'零':0,u'壹':1,u'贰':2,u'叁':3,u'肆':4,u'伍':5,u'陆':6,u'柒':7,u'捌':8,u'玖':9,
            u'〇':0,u'一':1,u'二':2,u'三':3,u'四':4,u'五':5,u'六':6,u'七':7,u'八':8,u'九':9,
            u'两':2,u'俩':2}
    return switch[cnstr]

def clrmb2numrmb(clrmb):
    ''' chinese language RMB representation to number representation
        for example  叁仟肆佰伍拾贰万贰仟壹佰肆拾陆元捌角贰分
        range is 0.00--9999999999999999.99 壹亿亿元-壹分
    '''
    if clrmb.find(u'万亿')!=-1:
        l=clrmb.split(u'万亿')
        n1=clrmb2numrmb(l[0])
        if l[1]=='':
            n2=0
        else:
            n2=clrmb2numrmb(l[1])
        return n1*1000000000000+n2
    elif clrmb.find(u'亿')!=-1:
        l=clrmb.split(u'亿')
        n1=clrmb2numrmb(l[0])
        if l[1]=='':
            n2=0
        else:
            n2=clrmb2numrmb(l[1])
        return n1*100000000+n2
    elif clrmb.find(u'万')!=-1:
        l=clrmb.split(u'万')
        n1=clrmb2numrmb(l[0])
        if l[1]=='':
            n2=0
        else:
            n2=clrmb2numrmb(l[1])
        return n1*10000+n2
    elif clrmb.find(u'千')!=-1:
        l=clrmb.split(u'千')
        n1=clrmb2numrmb(l[0])
        if l[1]=='':
            n2=0
        else:
            n2=clrmb2numrmb(l[1])
        return n1*1000+n2
    elif clrmb.find(u'百')!=-1:
        l=clrmb.split(u'百')
        n1=clrmb2numrmb(l[0])
        if l[1]=='':
            n2=0
        else:
            n2=clrmb2numrmb(l[1])
        return n1*100+n2
    elif clrmb.find(u'十')!=-1:
        l=clrmb.split(u'十')
        if l[0]=='':
            l[0]=u'壹'
        n1=clrmb2numrmb(l[0])
        if l[1]=='':
            n2=0
        else:
            n2=clrmb2numrmb(l[1])
        return n1*10+n2
    elif clrmb.find(u'元')!=-1:
        l=clrmb.split(u'元')
        if l[0]=='':
            l[0]=u'零'
        if l[1]=='':
            l[1]=u'零'
        n1=clrmb2numrmb(l[0])
        n2=clrmb2numrmb(l[1])
        return n1+n2
    elif clrmb.find(u'角')!=-1:
        l=clrmb.split(u'角')
        if l[1]=='':
            l[1]=u'零'
        n1=clrmb2numrmb(l[0])
        n2=clrmb2numrmb(l[1])
        return n1*0.1+n2
    elif clrmb.find(u'分')!=-1:
        l=clrmb.split(u'分')
        return clrmb2numrmb(l[0])*0.01
    else :
        if len(clrmb)>len(u'零'):
            clrmb=clrmb.replace(u'零','') # 零玖 to 玖
        return cnstr2num(clrmb)

def chineseRMB2number(clrmb):
    clrmb=clrmb.replace(u'仟', u'千').replace(u'佰', u'百').replace(u'拾', u'十').replace(u'圆', u'元').replace(u'整', u'')
    if clrmb.find(u'元')==-1:
        clrmb=clrmb+u'元'
    return clrmb2numrmb(clrmb)

def normalize_money(txt):
    txt=txt.encode('utf-8').decode('utf-8')
    list_money = re.findall(re.compile(r'[\d\.，, ]+[百千万亿美欧日韩]*?元|[$￥][\d\.，, ]+|[零壹贰叁肆伍陆柒捌玖拾佰仟十百千万亿元角分]{3,}'.encode().decode()), txt)
    list_normalize_money=[]
    for money in list_money:
        money=money.replace(u'，',u'').replace(u',',u'').replace(u' ',u'')
        if u'零'in money or u'壹'in money or u'贰'in money or u'叁'in money or u'肆'in money or u'伍'in money or \
                        u'陆'in money or u'柒'in money or u'捌'in money or u'玖'in money or u'拾'in money: #汉语金额表示方法
            list_normalize_money.append([chineseRMB2number(money),u'元'])
        else:
            if u'千亿' in money:
                list_normalize_money.append([float(money.split(u'千亿')[0])*100000000000,money.split(u'千亿')[1]])
            elif u'百亿' in money:
                list_normalize_money.append([float(money.split(u'百亿')[0])*10000000000,money.split(u'百亿')[1]])
            elif u'十亿' in money:
                list_normalize_money.append([float(money.split(u'十亿')[0])*1000000000,money.split(u'十亿')[1]])
            elif u'亿' in money:
                list_normalize_money.append([float(money.split(u'亿')[0])*100000000,money.split(u'亿')[1]])
            elif u'千万' in money:
                list_normalize_money.append([int(money.split(u'千万')[0])*10000000,money.split(u'千万')[1]])
            elif u'百万' in money:
                list_normalize_money.append([float(money.split(u'百万')[0])*1000000,money.split(u'百万')[1]])
            elif u'十万' in money:
                list_normalize_money.append([float(money.split(u'十万')[0])*100000,money.split(u'十万')[1]])
            elif u'万' in money:
                list_normalize_money.append([float(money.split(u'万')[0])*10000,money.split(u'万')[1]])
            elif u'千' in money:
                list_normalize_money.append([float(money.split(u'千')[0])*1000,money.split(u'千')[1]])
            elif u'百' in money:
                list_normalize_money.append([float(money.split(u'百')[0])*100,money.split(u'百')[1]])
            elif u'十' in money:
                list_normalize_money.append([float(money.split(u'十')[0])*10,money.split(u'十')[1]])
            elif u'￥' in money:
                list_normalize_money.append([float(money.replace(u'￥',u'')), u'元'])
            elif u'$' in money:
                list_normalize_money.append([float(money.replace(u'$', u'')), u'美元'])
            else:
                list_normalize_money.append([money])

    for money,normalize_money in zip(list_money,list_normalize_money):
        if len(normalize_money)>1:
            number=round(normalize_money[0])
            if int(number)==number:
                number=int(number)
            txt=txt.replace(money,str(number)+normalize_money[1])
        else:
            txt = txt.replace(money, normalize_money[0])
    return txt


if __name__=='__main__':
    a0=chineseRMB2number('玖亿捌仟陆佰玖拾万陆仟肆佰肆拾捌元')
    a=chineseRMB2number(u'叁仟肆佰伍拾贰万贰仟壹佰肆拾陆元捌角贰分')
    a1 = chineseRMB2number(u'叁佰亿元')
    a2=chineseRMB2number(u'叁万亿元')
    b=chineseRMB2number(u'拾亿零壹佰壹拾万元')
    c=chineseRMB2number('拾元')
    d=chineseRMB2number('壹亿壹仟万元')
    e=chineseRMB2number('肆亿陆仟壹佰陆拾贰万捌仟贰佰伍拾捌元零玖分')
    f=chineseRMB2number('肆亿零壹佰陆拾贰万零伍拾捌元零玖分')
    g=chineseRMB2number('肆万亿零壹佰陆拾贰万零伍拾捌元零玖分')
    print(a,b)