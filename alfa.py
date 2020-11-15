import detectron2
import numpy as np
import cv2
import random
import time
import os
import vlc
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

setup_logger()

# import some common libraries
#from detectron2.evaluation.coco_evaluation import COCOEvaluator
# import some common detectron2 utilities




from PIL import Image, ImageDraw, ImageFont


	
from PIL import ImageChops

import os
duration = 1  # seconds
freq = 440  # Hz
from pygame import mixer  # Load the popular external library














register_coco_instances("my_dataset_train", {}, "train/_annotations.coco.json", "train/imagenes")
register_coco_instances("my_dataset_val", {}, "valid/_annotations.coco.json", "valid/imagenes")
register_coco_instances("my_dataset_test", {}, "test/_annotations.coco.json", "test/imagenes")
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")


from detectron2.utils.visualizer import Visualizer

a1=1
a2=2
b1=1
b2=2
cord1,cord2,ord1,ord2=100,200,100,200
valorgrisimagen=0
valorgrisimagen1=0
valorgrisimagen2=0
valorgrisimagen3=0
valorgrisimagen4=0
valorgrisimagen5=0
valorgrisimagenreserva=0
temperaturafinal=25


valorresguardoa1=3
valorresguardoa2=0
valorresguardoa3=0
valorresguardoab1=2
valorresguardoab2=0
valorresguardoab3=0
















el0 = cv2.imread('cero.png',0)
el1 = cv2.imread('uno.png',0)
el2 = cv2.imread('dos.png',0)
el3 = cv2.imread('tres.png',0)
el4 = cv2.imread('cuatro.png',0)
el5 = cv2.imread('cinco.png',0)
el6 = cv2.imread('seis.png',0)
el7 = cv2.imread('siete.png',0)
el8 = cv2.imread('ocho.png',0)
el9 = cv2.imread('nueve.png',0)



def elige(argument):
    flag0=False
    flag1=False
    flag2=False
    flag3=False
    flag4=False
    flag5=False
    flag6=False
    flag7=False
    flag8=False
    flag9=False
    

    
    if (argument==el0).all():
    	return 0
   
    	
    if (argument==el1).all():
    	return 1
   
    if (argument==el2).all():
    	return 2
    		
    if (argument==el3).all():
    	return 3
    if (argument==el4).all():
    	return 4
    if (argument==el5).all():
    	return 5
    	
    if (argument==el6).all():
    	return 6	
    if (argument==el8).all():
    	return 7	
    	
    if (argument==el7).all():
    	return 8	
    	
    if (argument==el9).all():
    	return 9	
    	
    	













class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05




cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = CocoTrainer(cfg)
#trainer.resume_or_load(resume=False)
#trainer.train()





from detectron2.utils.visualizer import ColorMode



cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("my_dataset_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_test")








cap = cv2.VideoCapture('1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    #texto = pytesseract.image_to_string(frame([cord2:ord2,cord1:ord1])
    
    
    numerograyout1=cv2.cvtColor(frame[8:22,280:316], cv2.COLOR_BGR2GRAY)
    numerograyout2=cv2.cvtColor(frame[219:233,280:316], cv2.COLOR_BGR2GRAY)
    

    digitoarriba1=numerograyout1[1:13,1:9]
    _, digitoarriba1 = cv2.threshold(digitoarriba1, 210, 255, cv2.THRESH_BINARY)
    
    
    digitoarriba2=numerograyout1[1:13,10:18]
    _, digitoarriba2 = cv2.threshold(digitoarriba2, 210, 255, cv2.THRESH_BINARY)
   
    
    digitoarriba3=numerograyout1[1:13,24:32]
    _, digitoarriba3 = cv2.threshold(digitoarriba3, 210, 255, cv2.THRESH_BINARY)
    
    
    digitoabajo1=numerograyout2[1:13,1:9]
    _, digitoabajo1 = cv2.threshold(digitoabajo1, 210, 255, cv2.THRESH_BINARY)
   
   
    digitoabajo2=numerograyout2[1:13,10:18]
    _, digitoabajo2 = cv2.threshold(digitoabajo2, 210, 255, cv2.THRESH_BINARY)
    
    
    digitoabajo3=numerograyout2[1:13,24:32]
    _, digitoabajo3 = cv2.threshold(digitoabajo3, 210, 255, cv2.THRESH_BINARY)






    escalagrayout=cv2.cvtColor(frame[31:211,308:314], cv2.COLOR_BGR2GRAY)
    
    
    
    
    
    outputs = predictor(frame)
    output_pred_boxes = outputs["instances"].pred_boxes
    for i in output_pred_boxes.__iter__():
    	#print(i.cpu().numpy())
    	
    	x=i.cpu().numpy()
    	a1=int(x[0])
    	a2=int(x[1])
    	b1=int(x[2])
    	b2=int(x[3])
    	cord1=a1
    	ord1=b1
    	cord2=a2
    	ord2=b2

       
        
   	
    	"""print("cord1 "+str(a1))
    	print("cord2 "+str(a2))
    	print("ord1 "+str(b1))
    	print("ord2 "+str(b2))"""
    
    v = Visualizer(frame[:, :, ::-1],
                metadata=test_metadata, 
                scale=1
                 )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    
    
    
    
    
    
    
    
    grayout= cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("image",grayout)	
        	
        	
        	
        	
    
    if(ord1!=100 and ord2!=200 and cord1!=100 and cord2!=200 and cord2!=3 and ord2!=25 and cord1!=23 and ord1!=44  ):
            	#print("a")
            	imagensalida=out.get_image()[:, :, ::-1]
            	#[ord1:ord2,cord1:cord2]
            	print(cord2)
            	print(ord2)
            	print(cord1)
            	print(ord1)
            	grayoutcortada= cv2.cvtColor(imagensalida[cord2:ord2,cord1:ord1], cv2.COLOR_BGR2GRAY)
            	
            	
            	#cv2.imshow("image2",grayoutcortada)
            	for i in range(len(grayoutcortada)):
            		
            		for j in range(int(len(grayoutcortada[0])/5)):
            			#print(grayoutcortada[i][j])
            			if grayoutcortada[i][j]>valorgrisimagen5:
            				valorgrisimagen5=grayoutcortada[i][j]
            				if grayoutcortada[i][j]>valorgrisimagen4:
            					valorgrisimagen4=grayoutcortada[i][j]
            					if grayoutcortada[i][j]>valorgrisimagen3:
            						valorgrisimagen3=grayoutcortada[i][j]
            						if grayoutcortada[i][j]>valorgrisimagen2:
            							valorgrisimagen2=grayoutcortada[i][j]
            							if grayoutcortada[i][j]>valorgrisimagen1:
            								valorgrisimagen1=grayoutcortada[i][j]
            

        	
        	
            	
            	valorgrisimagen=(valorgrisimagen1+valorgrisimagen2+valorgrisimagen3+valorgrisimagen4+valorgrisimagen5)
            	print(valorgrisimagen)
            	

            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	
            	cord1,cord2,ord1,ord2=100,200,100,200
    #print(outputs["instances"].pred_classes)
    #print(outputs["instances"].pred_boxes)
    #print(type(outputs["instances"].pred_boxes))
   
    

   

   
    
    #cv2.imshow("image3",numerograyout1)
   
    
    #cv2.imshow("image4",numerograyout2)
    
    #cv2.imshow("image5",escalagrayout)
    
    
    #cv2.imshow("image6",digitoarriba1)
    
 
 
 
 
    valordigitoarriba1=elige(digitoarriba1)
    if valordigitoarriba1!=None:
    	valorresguardoa1=valordigitoarriba1
    if valordigitoarriba1==None:
    	valordigitoarriba1=valorresguardoa1
    valordigitoarriba2=elige(digitoarriba2)
    if valordigitoarriba2!=None:
    	valorresguardoa2=valordigitoarriba2
    if valordigitoarriba2==None:
    	valordigitoarriba2=valorresguardoa2	
    valordigitoarriba3=elige(digitoarriba3)
    if valordigitoarriba3!=None:
    	valorresguardoa3=valordigitoarriba3
    if valordigitoarriba3==None:
    	valordigitoarriba3=valorresguardoa3	
    valordigitoabajo1=elige(digitoabajo1)
    if valordigitoabajo1!=None:
    	valorresguardoab1=valordigitoabajo1
    if valordigitoabajo1==None:
    	valordigitoabajo=valorresguardoab1	
    valordigitoabajo2=elige(digitoabajo2)
    if valordigitoabajo2!=None:
    	valorresguardoab2=valordigitoabajo2
    if valordigitoabajo2==None:  
    	valordigitoabajo2=valorresguardoab2	
    valordigitoabajo3=elige(digitoabajo3)
    if valordigitoabajo3!=None:
    	valorresguardoab3=valordigitoabajo3   
    if valordigitoabajo3==None:
    	valordigitoabajo3=valorresguardoab3	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    
    
    temperaturaarriba=(str(valordigitoarriba1)+str(valordigitoarriba2)+(".")+str(valordigitoarriba3))

    #print(float(temperaturaarriba))
    temperaturarribaint=float(temperaturaarriba)
    temperaturabajo=(str(valordigitoabajo1)+str(valordigitoabajo2)+(".")+str(valordigitoabajo3))
    temperaturaabajoint=float(temperaturabajo)
    #print(float(temperaturabajo))
   
    
    #cv2.imshow("image7",numerograyout1)
    #cv2.imshow("image8",numerograyout2)
    pendiente=(temperaturarribaint-temperaturaabajoint)/255
   

    if valorgrisimagen>10:
   	 

   	 temperaturafinal=temperaturaabajoint+pendiente*valorgrisimagen
   	 print(temperaturafinal)
    
   
   
    
    
    
    message=(str(temperaturafinal)+" GRADOS")
    
    
    
   
    
    
    
    #print(message)
    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # Use putText() method for 
    # inserting text on video 
    cv2.putText(grayoutcortada,  
                message,  
                (10, 10),  
                font, 0.4,
                  
                (255,0 , 0),  
                2,  
                cv2.LINE_4) 
    cv2.imshow("imagefinal",grayoutcortada)
    if(temperaturafinal)>31:
    	os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    	#os.system('spd-say "An infected has entered, run"')
    	#p = vlc.MediaPlayer("sonido.mp3")
    	#p.play()
    	
	
       


    
    
    
    """
    print("EL2----------------------------------")
    print(el2)
    print("----------------------------------")
    print(digitoabajo3)
    print(digitoabajo1)
    print(digitoabajo2)
    """
    
    """
    cv2.imshow("image7",digitoarriba2)
    
    cv2.imshow("image8",digitoarriba3)
    
    cv2.imshow("image9",digitoabajo1)
    
    cv2.imshow("image10",digitoabajo2)
    cv2.imshow("image11",digitoabajo3)"""
    
  
    
    
    
    #texto = pytesseract.image_to_string(frame[8:22,280:316])
    #texto1 = pytesseract.image_to_string(frame[218:235,280:316])

    # Mostramos el resultado
    #print("eo"+texto)
    #print(texto1)
    	
    valorgrisimagen,valorgrisimagen1,valorgrisimagen2,valorgrisimagen3,valorgrisimagen4,valorgrisimagen5=0,0,0,0,0,0
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()









