import sys
import os
import time
import datetime
import traceback
import gc
import requests
# from concurrent import futures
import random

import numpy as np
import numba as nb
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from FlagEmbedding import FlagModel, FlagReranker


# fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from _config import logger


os.environ["TOKENIZERS_PARALLELISM"] = "true"
output_model_dir = "./_output" # trained model


## FastAPI & CORS (Cross-Origin Resource Sharing) ##
app = FastAPI(
    title="bb8-nlu-embedder",
    version="0.2.5"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@nb.jit(nopython=True)
def numpy_to_list(vector: np.ndarray) -> list:
    n = vector.shape[0]  # Get the length of the 1D array
    result = [0.0] * n  # Initialize a list with float elements

    for i in range(n):
        result[i] = float(vector[i])  # Convert each element to float and assign to result list

    return result  # Return the result as a list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_properties = torch.cuda.get_device_properties(0)


## Load Models ##
nlu_embedder = SentenceTransformer('bespin-global/klue-sroberta-base-continue-learning-by-mnr', device=device)
nlu_embedder.to(device)
assist_bi_encoder = FlagModel('BAAI/bge-base-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation
assist_cross_encoder = FlagReranker('BAAI/bge-reranker-base', use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation


egg_plant_embedding = [
        -0.034370340406894684,
        -0.02726297453045845,
        -0.01965866982936859,
        -0.022362815216183662,
        0.03886841610074043,
        0.01894131302833557,
        0.00822541769593954,
        0.07428457587957382,
        -0.007561848033219576,
        -0.037335023283958435,
        -0.018411869183182716,
        -0.07597164064645767,
        -0.05744272097945213,
        0.004396981094032526,
        -0.021354300901293755,
        0.024321120232343674,
        -0.004778302740305662,
        0.05058139190077782,
        -0.04504441097378731,
        0.028986787423491478,
        -0.009211783297359943,
        0.01073568593710661,
        0.006612553261220455,
        0.04103605076670647,
        0.08146500587463379,
        -0.0527210459113121,
        -0.05139404907822609,
        0.030892496928572655,
        -0.0468984991312027,
        0.027308929711580276,
        -0.021390078589320183,
        0.037023674696683884,
        0.03234368562698364,
        -0.015734393149614334,
        0.0043045030906796455,
        -0.008458922617137432,
        -0.016346970573067665,
        -0.015910066664218903,
        -0.0026673777028918266,
        -0.0008777761831879616,
        0.0001497323246439919,
        0.0021332469768822193,
        0.028799142688512802,
        0.007826866582036018,
        0.030986756086349487,
        0.005635288078337908,
        -0.034643128514289856,
        0.06457463651895523,
        -0.06544498354196548,
        0.04961971938610077,
        -0.0022325655445456505,
        -0.030976250767707825,
        -0.0543895922601223,
        -0.04408368840813637,
        0.022990602999925613,
        0.08018945157527924,
        -0.04227675125002861,
        -0.05827911198139191,
        0.04756339266896248,
        -0.012367945164442062,
        -0.010324904695153236,
        0.040100764483213425,
        0.037797000259160995,
        -0.02802831307053566,
        0.024774672463536263,
        0.05710531026124954,
        -0.030787557363510132,
        0.006412438582628965,
        -0.04845380038022995,
        -0.016393402591347694,
        -0.04537783935666084,
        -0.02912316843867302,
        -0.03815271332859993,
        -0.013027365319430828,
        0.016839835792779922,
        -0.02487204037606716,
        -0.08063184469938278,
        0.03712213411927223,
        0.04603767395019531,
        0.035248834639787674,
        -0.02719404175877571,
        -0.0072322627529501915,
        -0.0036977254785597324,
        -0.016947457566857338,
        0.00822621863335371,
        0.004892395809292793,
        -0.017718270421028137,
        -0.02758091688156128,
        -0.012412674725055695,
        0.010305332019925117,
        0.01897820457816124,
        -0.05030784755945206,
        0.03373054787516594,
        0.01666819490492344,
        0.024835733696818352,
        -0.07500238716602325,
        0.05753663554787636,
        0.013302040286362171,
        0.023738864809274673,
        -0.0537932813167572,
        -0.03906050696969032,
        0.04576324298977852,
        0.008137064054608345,
        0.05568096786737442,
        -0.08699249476194382,
        -0.0022472296841442585,
        -0.04720013216137886,
        -0.0076775504276156425,
        -0.03653865307569504,
        -0.011623078025877476,
        -0.0435914471745491,
        0.035831209272146225,
        0.035749588161706924,
        -0.00011152661318192258,
        -0.01644895412027836,
        0.07920701801776886,
        0.026018427684903145,
        -0.04867304489016533,
        -0.01557343453168869,
        -0.03244616463780403,
        -0.003863245015963912,
        -0.016548728570342064,
        -0.0345914252102375,
        0.04272027313709259,
        0.0024749906733632088,
        0.04889295995235443,
        -0.051328323781490326,
        0.046844687312841415,
        0.008261042647063732,
        -0.034791797399520874,
        -0.030156349763274193,
        0.016851752996444702,
        0.03659292683005333,
        -0.022015277296304703,
        0.028171846643090248,
        0.011784600093960762,
        0.022695789113640785,
        -0.025118917226791382,
        0.07772313803434372,
        -0.015029429458081722,
        -0.06153348460793495,
        0.028626544401049614,
        0.021462902426719666,
        -0.02486760914325714,
        0.05220598354935646,
        0.011988364160060883,
        0.0027065956965088844,
        -0.08289939165115356,
        -0.015518397092819214,
        0.019486702978610992,
        0.012822533026337624,
        0.04381079226732254,
        -0.0037561797071248293,
        0.035831283777952194,
        -0.009926995262503624,
        -0.017490876838564873,
        0.0006183690275065601,
        -0.015636252239346504,
        -0.0024726351257413626,
        -0.032396283000707626,
        0.024829279631376266,
        0.04176171496510506,
        0.02511100471019745,
        0.05787722393870354,
        -0.013861602172255516,
        -0.04422856122255325,
        0.06576955318450928,
        -0.006795841734856367,
        -0.07435424625873566,
        -0.006702238693833351,
        -0.10006964206695557,
        -0.025030935183167458,
        0.08922655135393143,
        0.004944914020597935,
        0.038743384182453156,
        -0.027659479528665543,
        0.007067560683935881,
        -3.6428216844797134e-05,
        -0.006461070850491524,
        0.02459007129073143,
        -0.053324855864048004,
        0.024536680430173874,
        -0.00885691400617361,
        0.011462789960205555,
        -0.01283496804535389,
        -0.016031378880143166,
        0.049019668251276016,
        -0.004072731826454401,
        -0.01095041073858738,
        -0.03317239135503769,
        -0.006688199006021023,
        -0.049681950360536575,
        -0.016180697828531265,
        -0.0021445429883897305,
        -0.012662473134696484,
        -0.002493901178240776,
        -0.0432736761868,
        0.025529487058520317,
        0.010058493353426456,
        0.03038186952471733,
        -0.001493479241617024,
        -0.005748727358877659,
        -0.012752806767821312,
        -0.03993925452232361,
        -0.05615988373756409,
        -0.014692673459649086,
        0.008689911104738712,
        -0.004865591414272785,
        0.05700000375509262,
        0.017657896503806114,
        -0.040757883340120316,
        0.08335377275943756,
        0.028568724170327187,
        -0.011944837868213654,
        -0.010552510619163513,
        0.052359454333782196,
        0.006081451661884785,
        -0.013726838864386082,
        0.019719574600458145,
        -0.046744294464588165,
        -0.03804755583405495,
        0.04936104267835617,
        -0.015988385304808617,
        -0.033256348222494125,
        0.03415551781654358,
        0.04294758290052414,
        0.03557112067937851,
        -0.043580856174230576,
        0.006743027362972498,
        0.02064734324812889,
        -0.00699003878980875,
        0.003835729556158185,
        0.021044112741947174,
        -0.01415660697966814,
        -0.02923433668911457,
        -0.02090945467352867,
        0.05171149969100952,
        -0.01187245361506939,
        0.02439151331782341,
        -0.007591602858155966,
        0.002624707529321313,
        0.0683252215385437,
        -0.05947309732437134,
        -0.010554865933954716,
        -0.0277395136654377,
        0.012483588419854641,
        -0.010624282993376255,
        -0.006776363588869572,
        -0.03741571679711342,
        -0.016572361811995506,
        -0.013713888823986053,
        0.021652884781360626,
        0.02634989283978939,
        0.02732199989259243,
        0.012017396278679371,
        -0.031922440975904465,
        -0.009502407163381577,
        0.0357518345117569,
        0.04960784688591957,
        0.02894497849047184,
        -0.02172108180820942,
        -0.0034443032927811146,
        0.024697622284293175,
        0.03069857694208622,
        -0.02149983122944832,
        -0.023989783599972725,
        -0.04014900326728821,
        -0.004525105468928814,
        0.042494162917137146,
        -0.03885861858725548,
        0.07712960243225098,
        0.0008217221475206316,
        -0.055754534900188446,
        0.005057806149125099,
        0.03461780026555061,
        0.008310206234455109,
        -0.09178265184164047,
        0.010565504431724548,
        -0.03727979585528374,
        -0.0018825214356184006,
        -0.02988177351653576,
        0.05762932077050209,
        -0.04260195419192314,
        -0.012574329972267151,
        0.028270695358514786,
        0.009743009693920612,
        -0.013602991588413715,
        -0.02321656234562397,
        0.012989888899028301,
        0.04340868443250656,
        0.01734236069023609,
        0.015091910026967525,
        -0.04777871072292328,
        0.028720416128635406,
        -0.016741838306188583,
        0.005285376682877541,
        0.05725817754864693,
        0.05773616582155228,
        -0.0015095509588718414,
        0.026250483468174934,
        0.007332589477300644,
        0.006154158618301153,
        -0.015308063477277756,
        -0.02502390928566456,
        0.0383668914437294,
        0.034588832408189774,
        0.039358485490083694,
        -0.0018070697551593184,
        -0.22380520403385162,
        0.018598712980747223,
        0.011944117024540901,
        0.023731257766485214,
        0.05719047412276268,
        -0.002307317452505231,
        0.0008863791008479893,
        -0.019902987405657768,
        0.012122156098484993,
        0.012574002146720886,
        -0.04168863967061043,
        -0.04119081795215607,
        0.03947025164961815,
        -0.012099800631403923,
        0.028726348653435707,
        0.037267789244651794,
        0.024718962609767914,
        -0.016152899712324142,
        0.004022833425551653,
        0.005316555965691805,
        -0.020897645503282547,
        -0.026013923808932304,
        0.008967973291873932,
        0.11294828355312347,
        -0.0015933880349621177,
        0.018025634810328484,
        0.0002889930037781596,
        0.01055680587887764,
        -0.006809040438383818,
        -0.011877449229359627,
        -0.003489929251372814,
        -0.04182657599449158,
        0.03777090087532997,
        0.022195015102624893,
        -0.022714635357260704,
        0.009823549538850784,
        -0.010861986316740513,
        -0.026428241282701492,
        0.05445411801338196,
        0.01159584615379572,
        -0.02058623544871807,
        -0.04138484597206116,
        0.01756168343126774,
        0.034315936267375946,
        -0.006878786254674196,
        -0.045756999403238297,
        -0.04489726573228836,
        -0.025186453014612198,
        0.032373182475566864,
        0.09363805502653122,
        0.018009033054113388,
        0.029437905177474022,
        -0.005060976836830378,
        -0.04209326580166817,
        -0.0125172920525074,
        0.012545475736260414,
        -0.014214698225259781,
        0.010462773963809013,
        0.013229209929704666,
        0.006425021216273308,
        0.00029070311575196683,
        0.018111208453774452,
        -0.08937147259712219,
        -0.00016057040193118155,
        0.056298382580280304,
        -0.050068579614162445,
        -0.07730218023061752,
        0.030347395688295364,
        0.035565443336963654,
        -0.04087032377719879,
        -0.020021701231598854,
        -0.036524105817079544,
        -0.06918422132730484,
        -0.10381032526493073,
        -0.024958664551377296,
        -0.03333284705877304,
        -0.07487213611602783,
        -0.009511049836874008,
        -0.009585017338395119,
        -0.01973132975399494,
        0.007357086054980755,
        0.048740193247795105,
        0.03921358659863472,
        -0.012909854762256145,
        -0.01547286007553339,
        0.00031481526093557477,
        -0.03200536593794823,
        0.009766132570803165,
        -0.028800370171666145,
        -0.03270404785871506,
        -0.01152811199426651,
        0.0016882624477148056,
        0.013261503539979458,
        -0.017030825838446617,
        0.015337906777858734,
        0.054905395954847336,
        -0.016942773014307022,
        0.012048511765897274,
        0.03522675856947899,
        0.014020698145031929,
        -0.005771036259829998,
        -0.029043495655059814,
        -0.036138344556093216,
        0.01677323691546917,
        0.05396943911910057,
        -0.0034395710099488497,
        -0.053168442100286484,
        0.020469311624765396,
        0.021801499649882317,
        -0.024509478360414505,
        0.015351016074419022,
        -0.04196290671825409,
        0.0061345831491053104,
        -0.05735389143228531,
        -0.019081024453043938,
        0.03240899369120598,
        0.026794640347361565,
        0.021867094561457634,
        0.09453943371772766,
        -0.03989351540803909,
        -0.024172727018594742,
        0.005171516444534063,
        -0.048856351524591446,
        -0.029531437903642654,
        0.0023350210394710302,
        -0.05425570532679558,
        0.015688661485910416,
        0.013271122239530087,
        -0.04795687273144722,
        -0.04382018372416496,
        0.010269560851156712,
        -0.030602751299738884,
        0.01899741031229496,
        0.06227065622806549,
        -0.04200122877955437,
        -0.03382491692900658,
        -0.007561274338513613,
        -0.037809453904628754,
        -0.07030418515205383,
        0.05115357041358948,
        0.027227673679590225,
        0.023610683158040047,
        0.0534026101231575,
        -0.004462398122996092,
        0.017815593630075455,
        0.007768325507640839,
        0.028093894943594933,
        -0.013456239365041256,
        -0.02823992259800434,
        0.021187853068113327,
        0.0672447457909584,
        0.07915731519460678,
        0.037841007113456726,
        -0.02011142671108246,
        -0.028485633432865143,
        -0.03794325888156891,
        0.016072921454906464,
        -0.007889598608016968,
        -0.08312539011240005,
        -0.06966761499643326,
        0.04171324893832207,
        -0.07246506959199905,
        0.05886557698249817,
        0.020657304674386978,
        -0.030564086511731148,
        0.06126375496387482,
        0.08760546892881393,
        -0.01403044257313013,
        -0.016996080055832863,
        -0.0205901600420475,
        0.06135221943259239,
        -0.043910738080739975,
        -0.026656517758965492,
        -0.04108697921037674,
        -0.044081006199121475,
        0.04323994368314743,
        0.016213931143283844,
        -0.0075555783696472645,
        0.022712690755724907,
        6.395440141204745e-05,
        -0.035250820219516754,
        -0.00772268557921052,
        0.00838520098477602,
        0.01469570305198431,
        0.06412862241268158,
        -0.012204550206661224,
        -0.008325982838869095,
        -0.04350313916802406,
        0.013498915359377861,
        -0.02350037544965744,
        -0.022374210879206657,
        0.04897478595376015,
        -0.011190236546099186,
        0.008996940217912197,
        -0.05069116875529289,
        -0.05689888447523117,
        -0.06302204728126526,
        -0.005399039946496487,
        0.0006691308226436377,
        -0.01108685415238142,
        -0.015905076637864113,
        -0.08131247013807297,
        -0.0037054771091789007,
        -0.009277141653001308,
        0.07652711123228073,
        -0.04528198763728142,
        0.011975760571658611,
        0.04942428320646286,
        0.015056769363582134,
        0.04042677953839302,
        0.02129076048731804,
        -0.06842902302742004,
        -0.007093979045748711,
        -0.05233238637447357,
        0.02288205921649933,
        0.021446451544761658,
        -0.04886618256568909,
        -0.03891841322183609,
        -0.0051559642888605595,
        0.014311720617115498,
        -0.026306580752134323,
        -0.0019070167327299714,
        0.024661943316459656,
        -0.0021989860106259584,
        0.020106082782149315,
        0.03824733942747116,
        0.010472958907485008,
        -0.006504969205707312,
        0.009531931951642036,
        -0.014398902654647827,
        0.014054272323846817,
        -0.03716570511460304,
        0.03614500164985657,
        -0.0034409710206091404,
        0.04560977965593338,
        -0.014437946490943432,
        0.01629573665559292,
        0.01014972198754549,
        0.06560119986534119,
        0.05184653401374817,
        0.033683303743600845,
        -0.008415041491389275,
        -0.006652800366282463,
        -0.022438697516918182,
        0.01353535521775484,
        -0.023263830691576004,
        -0.0009786204900592566,
        0.024149911478161812,
        -0.005178018473088741,
        0.022016426548361778,
        -0.014353781938552856,
        -0.06382830440998077,
        0.012236175127327442,
        0.0002785801771096885,
        0.011713352054357529,
        -0.10726838558912277,
        -0.05513869225978851,
        0.06982699781656265,
        -0.021110542118549347,
        0.05098419263958931,
        0.026835566386580467,
        0.022644584998488426,
        0.008462521247565746,
        -0.007238233461976051,
        0.04468974098563194,
        -0.003492828691378236,
        0.02187071368098259,
        -0.019098950549960136,
        0.029190003871917725,
        -0.0568401999771595,
        0.006049144547432661,
        -0.08815325051546097,
        -0.004844650626182556,
        0.0295129232108593,
        0.003966107033193111,
        -0.04796997457742691,
        -0.00656540784984827,
        -0.07397717982530594,
        0.031338028609752655,
        -0.01797155849635601,
        -0.03813423588871956,
        -0.0337865948677063,
        -0.01044397708028555,
        -0.033200785517692566,
        0.030897611752152443,
        -0.054791439324617386,
        -0.03691484034061432,
        -0.025659846141934395,
        -0.03078358992934227,
        -0.04118970409035683,
        -0.014361015520989895,
        0.06555459648370743,
        0.06708496809005737,
        0.010802408680319786,
        -0.00598742114380002,
        0.039521168917417526,
        0.025898637250065804,
        0.01833947002887726,
        -0.0154526736587286,
        -0.02561011165380478,
        -0.06299556791782379,
        0.020988505333662033,
        0.0361156202852726,
        0.002243105787783861,
        -0.033587127923965454,
        0.00686293002218008,
        -0.056016333401203156,
        0.002253062091767788,
        0.01843286119401455,
        -0.019641710445284843,
        0.021503474563360214,
        -0.02236008830368519,
        0.012386519461870193,
        0.01188711542636156,
        -0.0501476489007473,
        0.041614823043346405,
        -0.032841265201568604,
        -0.014422660693526268,
        0.04938136786222458,
        0.03197871893644333,
        0.03185553848743439,
        -0.014129218645393848,
        0.02466222085058689,
        -0.011253376491367817,
        0.07681535184383392,
        0.03378815948963165,
        0.01194407045841217,
        0.0066815209574997425,
        -0.01941106654703617,
        -0.007575227878987789,
        0.04733332619071007,
        0.015759682282805443,
        -0.03681465610861778,
        0.04257673770189285,
        -0.02412162534892559,
        -0.04662415385246277,
        -0.056816764175891876,
        -0.050537798553705215,
        -0.020042987540364265,
        0.03831329569220543,
        0.02616002969443798,
        0.030593398958444595,
        0.040290072560310364,
        -0.004053025972098112,
        0.04223022982478142,
        -0.03197900950908661,
        0.03641989827156067,
        0.006877685431391001,
        0.010129599831998348,
        0.023563235998153687,
        0.019131943583488464,
        -0.03524386137723923,
        0.02748178504407406,
        -0.05576726794242859,
        -0.02627910114824772,
        -0.033210013061761856,
        -0.059309449046850204,
        0.015093003399670124,
        -0.013303536921739578,
        -0.013842316344380379,
        -0.005444989539682865,
        -0.027979033067822456,
        0.05454395338892937,
        -0.031041204929351807,
        -0.02572173997759819,
        0.06630130112171173,
        -0.013658636249601841,
        -0.019371377304196358,
        0.020217405632138252,
        0.03126152232289314,
        0.06034768000245094,
        -0.04886240139603615,
        0.016469724476337433,
        -0.0033748229034245014,
        0.013096078298985958,
        0.000999409705400467,
        0.02869437448680401,
        -0.04929208382964134,
        0.016240200027823448,
        0.03187816962599754,
        -0.0349159874022007,
        -0.05583489313721657,
        -0.023625573143363,
        -0.036242466419935226,
        -0.021390194073319435,
        -0.02950354479253292,
        0.04203568398952484,
        0.02835937589406967,
        -0.07116998732089996,
        -0.007673575077205896,
        0.023489829152822495,
        -0.04243402183055878,
        0.04369523003697395,
        0.01815524697303772,
        -0.024390997365117073,
        0.04702814295887947,
        0.0006349179311655462,
        0.05075938627123833,
        0.01067526638507843,
        -0.013426664285361767,
        0.07429391890764236,
        0.026256747543811798,
        -0.020693179219961166,
        -0.029044032096862793,
        0.010084918700158596,
        0.055523041635751724,
        -0.0005572437075898051,
        -0.004218115471303463,
        -0.05128210037946701,
        0.039693739265203476,
        0.046642426401376724,
        0.01246243342757225,
        -0.026657413691282272,
        -0.011949599720537663,
        -0.024403618648648262,
        -0.013121254742145538,
        0.01507980190217495,
        0.014270001091063023,
        -0.010835695080459118,
        -0.008671470917761326,
        -0.026642316952347755,
        0.020526662468910217,
        -0.04946092516183853,
        0.029348954558372498,
        -0.07120804488658905,
        -0.0006024806061759591,
        0.03259816765785217,
        0.06208229437470436,
        -0.04204350709915161,
        0.026230860501527786,
        0.04731667414307594,
        0.008753982372581959,
        -0.028575580567121506,
        0.009566051885485649,
        -0.017183061689138412,
        -0.012505028396844864,
        0.035016268491744995,
        0.009489630348980427,
        -0.04386789724230766,
        -0.02983967773616314,
        0.03873898461461067,
        0.03847057372331619,
        -0.0025553377345204353,
        -0.021030131727457047,
        0.035454098135232925,
        -0.006131754722446203,
        0.032532699406147,
        -8.672817784827203e-05,
        -0.04452946037054062,
        0.042122311890125275,
        0.004880252759903669,
        0.005155010148882866,
        -0.0062813470140099525,
        0.0032680677250027657,
        0.011996866203844547,
        -0.004174345638602972,
        0.0465996190905571,
        0.02272183448076248,
        0.017555290833115578,
        0.0047440784983336926
    ]
@app.get('/health')
def health_check():
    '''
    Health Check
    '''
    return JSONResponse({'status':"helpnow-embedder is listening..", "timestamp":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})


class EmbeddingItem(BaseModel):
    data = []

@app.get("/api/nlu/sentence-embedding")
def sentence_embedding(query):
    try:
        embed_vector = nlu_embedder.encode(query, device=device)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vector = None

    #embed_vector = [float(v) for v in embed_vector]

    return JSONResponse({'embed_vector': numpy_to_list(embed_vector)})


@app.post("/api/nlu/sentence-embedding-batch")
def sentence_embedding_batch(item: EmbeddingItem):
    item = item.dict()
    data = item['data']
    query_list = [r['text'] for r in data]

    try:
        embed_vectors = nlu_embedder.encode(query_list, device=device)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vectors = None

    for i, row in enumerate(data):
        row['embed_vector'] = [float(v) for v in embed_vectors[i]]

    return JSONResponse(data)


@app.get("/api/assist/sentence-embedding")
def sentence_embedding(query: str):
    # try:
    #     embed_vector = assist_bi_encoder.encode_queries(query) # query_instruction_for_retrieval + query
    # except:
    #     logger.error(f'{traceback.format_exc()}')
    #     embed_vector = None

    #embed_vector = [float(v) for v in embed_vector]

    return JSONResponse({'embed_vector': egg_plant_embedding})


@app.post("/api/assist/sentence-embedding-batch")
def sentence_embedding_batch(item: EmbeddingItem):
    item = item.dict()
    data = item['data']
    query_list = [r['text'] for r in data]

    try:
        embed_vectors = assist_bi_encoder.encode(query_list)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vectors = None

    for i, row in enumerate(data):
        row['embed_vector'] = numpy_to_list(embed_vectors[i])

    return JSONResponse(data)


@app.post("/api/assist/cross-encoder/similarity-scores")
def sentence_embedding_batch(item: EmbeddingItem):
    s = time.time()
    item = item.dict()
    data = item['data']
    #
    # query_doc_list = [[r['query'], r['passage']] for r in data]
    # similarity_scores = assist_cross_encoder.compute_score(query_doc_list)
    # print(f'⏱️ process time of cross-encoder: {time.time() - s}')

    return JSONResponse({"similarity_scores": [random.uniform(-10, 10) for _ in data]})


#===========================
# CUDA Memory Check 스케쥴러 입니다.
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

def check_cuda_memory():
    if device.type == "cuda":
        current_memory = round(torch.cuda.memory_allocated() / (1024 ** 3), 4)
        total_memory = round(gpu_properties.total_memory / (1024 ** 3), 4)
        print(f'>> Usage of Current Memory: {current_memory} GB / {total_memory} GB')

        gc.collect()
        torch.cuda.empty_cache()
    else:
        print('>> Not using CUDA.')

scheduler = BackgroundScheduler()

# 스케줄러에 작업 추가 (예: 10초마다 실행)
scheduler.add_job(check_cuda_memory, IntervalTrigger(seconds=30))
# 스케줄러 시작
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()