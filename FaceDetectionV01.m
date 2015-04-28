% detecte les différentes parties du visage telles que la bouche, le nez, les yeux

clear

faceDetector = vision.CascadeObjectDetector;

famille=imread('famille.jpg');
% rotation à faire pour améliorer les résultats
% famille = imrotate(famille1,8,'bicubic');
Face = step(faceDetector, famille);
% IFaces = insertObjectAnnotation(famille, 'rectangle', bboxes, 'Face');
imgFace = (famille(Face(1,2):Face(1,2)+Face(1,4),Face(1,1):Face(1,1)+Face(1,3),:));
figure, imshow(famille), title('original');
% figure, imshow(imgFace), title('Detected faces');
[x,y,z]=size(famille);
[h,w,dim]=size(imgFace);
% [height, width, dim] = size(image);
% [x y width height], that defines a rectangular region of interest within image I
   
eyeLeftDetector = vision.CascadeObjectDetector('LeftEye','UseROI',true);
% Search in upper left corner
RoieyeLeft = [w/7, 1, w*2/7, h/2]; 
bboxEyeLeft = step(eyeLeftDetector, imgFace, RoieyeLeft);
ILeftEye = insertObjectAnnotation(imgFace, 'rectangle',bboxEyeLeft,'LeftEye');
% figure, imshow(ILeftEye), title('Detected LeftEye');

eyeRightDetector = vision.CascadeObjectDetector('RightEye','UseROI',true);
% Search in upper right corner
RoieyeRight = [w*4/7, 1, w*2/7, h/2]; 
bboxEyeRight = step(eyeRightDetector, imgFace, RoieyeRight);
IRightEye = insertObjectAnnotation(imgFace, 'rectangle',bboxEyeRight,'RightEye');
% figure, imshow(IRightEye), title('Detected RightEye');

mouthDetector = vision.CascadeObjectDetector('Mouth','UseROI',true);
% Search in below part of the image
RoiMouth = [w/6, h*2/3, w*2/3, h/3]; 
bboxMouth = step(mouthDetector, imgFace,RoiMouth);
IMouth = insertObjectAnnotation(imgFace, 'rectangle',bboxMouth,'Mouth');
% figure, imshow(IMouth), title('Detected Mouth');

mouthDetector = vision.CascadeObjectDetector('Nose','UseROI',true);
% Search in middle part of the image
RoiNose = [w/4, h/4, w/2, h/2]; 
bboxNose = step(mouthDetector, imgFace,RoiNose);
INose = insertObjectAnnotation(imgFace, 'rectangle',bboxNose,'Nose');
% figure, imshow(INose), title('Detected Nose');

IFinalFace = insertObjectAnnotation(imgFace, 'rectangle',bboxEyeLeft,'LeftEye');
IFinalFace1 = insertObjectAnnotation(IFinalFace, 'rectangle',bboxEyeRight,'RightEye');
IFinalFace2 = insertObjectAnnotation(IFinalFace1, 'rectangle',bboxMouth,'Mouth');
IFinalFace3 = insertObjectAnnotation(IFinalFace2, 'rectangle',bboxNose,'Nose');
figure, imshow(IFinalFace3), title('FeatureDetected');
