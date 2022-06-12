clc;
close all;
filename = imread('C:\Users\vasu\Desktop\datasets\raw-890\42_img_.png'); 

%%%%%%%%%%%%%%%%%%%%   WHITE BALANCING   %%%%%%%%%%%%%%%%%%%%%%%
% We consider input1 as White_Balanced image and input2 as DCP
% Reading the input image with imread   rawimages\6_img_.png   refimages\6_img_.png
%(im2double) function is used for precision by converting image from uint8 into double format
RGB_Image_o = (filename);
Ref_Image = (filename);
RGB_Image = im2double(RGB_Image_o);

%Channel separation
Red_Channel = RGB_Image(:, :, 1);
Green_Channel = RGB_Image(:, :, 2);
Blue_Channel = RGB_Image(:, :, 3);

%Sum of each color channel
R = sum(Red_Channel,'all');
G = sum(Green_Channel,'all');
B = sum(Blue_Channel,'all');

%Finding mean of each color channel
mean_Red = mean2(Red_Channel);
mean_Green = mean2(Green_Channel);
mean_Blue = mean2(Blue_Channel);

%Calculating RGB gain factor
Lr = max(Red_Channel,[],'all');
Lg = max(Green_Channel,[],'all');
Lb = max(Blue_Channel,[],'all');

%Color estimation
L2=0.2;
Ur =(Lr*(R/mean_Red))+L2 ;
Ug =(Lg*(G/mean_Green))+L2 ;
Ub =(Lb*(B/mean_Blue))+L2 ;

%Color correction
r = Red_Channel/Ur;
g = Green_Channel/Ug; 
b = Blue_Channel/Ub;
img1=cat(3,r,g,b);         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Color      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Scaling
%mat2gray( A ) converts the matrix A to a grayscale image I 
%that contains values in the range 0 to 1 
r = uint8(255 * mat2gray(r));
g = uint8(255 * mat2gray(g));
b = uint8(255 * mat2gray(b));
white_balance = cat(3,r,g,b);     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%    white Balance      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imshowpair(RGB_Image_o,white_balance,'montage');
title('1. ORIGINAL (left) and it''s WHITE BALANCED(right) Image');


entro=entropy(RGB_Image_o);
fprintf('The entropy value for White Balanced Image is %f\n',entro);
MSE = immse(white_balance,Ref_Image);
fprintf('The MSE value for White Balanced Image is %0.4f\n',MSE);
PSNR = psnr(white_balance,Ref_Image);
fprintf('The PSNR value for White Balanced Image is %0.4f\n', PSNR);
SSIM = ssim(white_balance,Ref_Image);
fprintf('The SSIM value for White Balanced Image is %0.4f\n',SSIM);
uicm= UICM(RGB_Image_o);
fprintf('The UICM value for White Balanced Image is %f\n',uicm);
uism=UISM(RGB_Image_o);
fprintf('The UISM value for White Balanced Image is %f\n',uism);
uiconm=UIConM(RGB_Image_o);
fprintf('The UIConM value for White Balanced Image is %f\n',uiconm);
uiqm = UIQM(RGB_Image_o);
fprintf('The UIQM value for White Balanced Image is %f\n',uiqm);    
uciqe=UCIQE(RGB_Image_o);
fprintf('The UCIQE value for White Balanced Image is %f\n\n',uciqe);

%%%%%%%%%%%%%%%%%%%%%%%%%%  DARK CHANNEL PRIOR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hazy_image = white_balance;
% hazy_image = im2double(hazy_image);
%[J,T,L] = imreducehaze(I) reduces atmospheric haze in color or grayscale image I.
%The function returns the dehazed image J, an estimate T of the haze thickness at each pixel,
%and the estimated atmospheric light L.
%[1] He, Kaiming. "Single Image Haze Removal Using Dark Channel Prior." 
[dehazed_image,t,A]=imreducehaze(hazy_image,'method','approxdcp');


figure,imshow(dehazed_image);    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%    DCP      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imshowpair(white_balance,dehazed_image,'montage')
title('2.white balanced (left) and DCP(right) Image');
img2=im2double(dehazed_image);   
figure;imshow(img2);title("color");

img3=clahe(img2);
figure;imshow(img3);title('clahe vasu');
img33=im2uint8(img3);

entro1=entropy(dehazed_image);
fprintf('The entropy value for dehazed_image is %f\n',entro1);
MSE1 = immse(dehazed_image,Ref_Image);
fprintf('The MSE value for DCP Image is %0.4f\n',MSE1);
PSNR1 = psnr(dehazed_image,Ref_Image);
fprintf('The PSNR value for DCP Image is %0.4f\n', PSNR1);
SSIM1 = ssim(dehazed_image,Ref_Image);
fprintf('The SSIM value for DCP Image is %0.4f\n',SSIM1);
uicm1= UICM(dehazed_image);
fprintf('The UICM value for DCP Image is %f\n',uicm1);
uism1=UISM(dehazed_image);
fprintf('The UISM value for DCP Image is %f\n',uism1);
uiconm1=UIConM(dehazed_image);
fprintf('The UIConM value for DCP Image is %f\n',uiconm1);
uiqm1 = UIQM(dehazed_image);
fprintf('The UIQM value for DCP Image is %f\n',uiqm1);
uciqe1=UCIQE(dehazed_image);
fprintf('The UCIQE value for DCP Image is %f\n\n',uciqe1);



entro2=entropy(img33);
fprintf('The entropy value for CLAHE is %f\n',entro2);
MSE2 = immse(img33,Ref_Image);
fprintf('The MSE value for CLAHE is %0.4f\n',MSE2);
PSNR2 = psnr(img33,Ref_Image);
fprintf('The PSNR value for CLAHE is %0.4f\n', PSNR2);
SSIM2 = ssim(img33,Ref_Image);
fprintf('The SSIM value for CLAHE is %0.4f\n',SSIM2);
uicm2= UICM(img33);
fprintf('The UICM value for CLAHE is %f\n',uicm2);
uism2=UISM(img33);
fprintf('The UISM value for CLAHE is %f\n',uism2);
uiconm2=UIConM(img33);
fprintf('The UIConM value for CLAHE is %f\n',uiconm2);
uiqm2 = UIQM(img33);
fprintf('The UIQM value for CLAHE is %f\n',uiqm2);
uciqe2=UCIQE(img33);
fprintf('The UCIQE value for CLAHE is %f\n\n',uciqe2);

%%%%%%%%%%%%%% weight maps %%%%%%%%%%%%
% for input1
lab1 = rgb2lab(img1);
R1 = lab1(:, :, 1);

% calculate Global contrast weight
WG1 = abs(imfilter(R1, fspecial('Laplacian'), 'replicate', 'conv'));

%calculate Local contrast weight
h = 1/16* [1, 4, 6, 4, 1];
WL1 = imfilter(R1,  h'*h, 'replicate', 'conv');
WL1(WL1 > (180/2.75)) = 180/2.75;
WL1 = (R1 - WL1).^2;

% calculate the Saliency weight
WS1 = saliency_detection(img1);

% calculate the Exposedness weight
sigma = 0.25;
aver = 0.5;
WE1 = exp(-(R1 - aver).^2 / (2*sigma^2));

figure,montage({uint8(255*mat2gray(WG1)),uint8(255*mat2gray(WL1)),uint8(255*mat2gray(WS1)),uint8(255*mat2gray(WE1))})
title('WG1,WL1,WS1,WE1 (left-->right)')


%for input2
lab2 = rgb2lab(img2);
R2 = lab2(:, :, 1);

% calculate Global Contrast weight
WG2 = abs(imfilter(R2, fspecial('Laplacian'), 'replicate', 'conv'));

%calculate Local Contrast weight
h = 1/16* [1, 4, 6, 4, 1];
WL2 = imfilter(R2, h'*h, 'replicate', 'conv');
WL2(WL2 > (180/2.75)) = 180/2.75;
WL2 = (R2 - WL2).^2;

% calculate the Saliency weight
WS2 = saliency_detection(img2);
% calculate the Exposedness weight
sigma = 0.25;
aver = 0.5;
WE2 = exp(-(R2 - aver).^2 / (2*sigma^2));

figure,montage({uint8(255*mat2gray(WG2)),uint8(255*mat2gray(WL2)),uint8(255*mat2gray(WS2)),uint8(255*mat2gray(WE2))})
title('WG2,WL2,WS2,WE2 (left-->right)')

%%%%%%%%%%%%%%%%%%%%%%%input3
lab3 = rgb2lab(img1);
R3 = lab3(:, :, 1);

% calculate Global contrast weight
WG3 = abs(imfilter(R3, fspecial('Laplacian'), 'replicate', 'conv'));
%calculate Local contrast weight
h = 1/16* [1, 4, 6, 4, 1];
WL3 = imfilter(R3,  h'*h, 'replicate', 'conv');
WL3(WL3 > (180/2.75)) = 180/2.75;
WL3 = (R3 - WL3).^2;

% calculate the Saliency weight
WS3 = saliency_detection(img3);

% calculate the Exposedness weight
sigma = 0.25;
aver = 0.5;
WE3= exp(-(R3 - aver).^2 / (2*sigma^2));

figure,montage({uint8(255*mat2gray(WG3)),uint8(255*mat2gray(WL3)),uint8(255*mat2gray(WS3)),uint8(255*mat2gray(WE3))})
title('WG3,WL3,WS3,WE3 (left-->right)')




% calculate the normalized weight
W1 = (WG1 + WL1 + WS1 + WE1) ./ (WG1 + WL1 + WS1 + WE1 + WG2 + WL2 + WS2 + WE2+WG3 + WL3 + WS3 + WE3);
W2 = (WG2 + WL2 + WS2 + WE2) ./ (WG1 + WL1 + WS1 + WE1 + WG2 + WL2 + WS2 + WE2+WG3 + WL3 + WS3 + WE3);
W3 = (WG3 + WL3 + WS3 + WE3) ./ (WG1 + WL1 + WS1 + WE1 + WG2 + WL2 + WS2 + WE2+WG3 + WL3 + WS3 + WE3);
figure,montage({uint8(255*mat2gray(W1)),uint8(255*mat2gray(W2)),uint8(255*mat2gray(W3))});
title('W1,W2,W3 (left-->right)')

% calculate the gaussian pyramid for Normalised Weights
level = 5;
Weight1 = gaussian_pyramid(W1, level);
Weight2 = gaussian_pyramid(W2, level);
Weight3 = gaussian_pyramid(W3, level);
figure,montage({uint8(255*mat2gray(Weight1{1,1})),uint8(255*mat2gray(Weight1{1,2})),uint8(255*mat2gray(Weight1{1,3})),uint8(255*mat2gray(Weight1{1,4}))})
title('Gaussian Pyramid for Normalised Weight_1 (level 1-->level 4)')
figure,montage({uint8(255*mat2gray(Weight2{1,1})),uint8(255*mat2gray(Weight2{1,2})),uint8(255*mat2gray(Weight2{1,3})),uint8(255*mat2gray(Weight2{1,4}))})
title('Gaussian Pyramid for Normalised Weight_2 (level 1-->level 4)')  
figure,montage({uint8(255*mat2gray(Weight3{1,1})),uint8(255*mat2gray(Weight3{1,2})),uint8(255*mat2gray(Weight3{1,3})),uint8(255*mat2gray(Weight3{1,4}))})
title('Gaussian Pyramid for Normalised Weight_2 (level 1-->level 4)') 

% calculate the laplacian pyramid for input1 and input2
% input1
R1 = laplacian_pyramid(img1(:, :, 1), level);
G1 = laplacian_pyramid(img1(:, :, 2), level);
B1 = laplacian_pyramid(img1(:, :, 3), level);
figure,montage({uint8(255*mat2gray(R1{1,1})),uint8(255*mat2gray(R1{1,2})),uint8(255*mat2gray(R1{1,3})),uint8(255*mat2gray(R1{1,4}))})
title('Laplacian Pyramid for Input 1(WB) (level 1-->level 4)')  

% input2
R2 = laplacian_pyramid(img2(:, :, 1), level);
G2 = laplacian_pyramid(img2(:, :, 2), level);
B2 = laplacian_pyramid(img2(:, :, 3), level);
figure,montage({uint8(255*mat2gray(R2{1,1})),uint8(255*mat2gray(R2{1,2})),uint8(255*mat2gray(R2{1,3})),uint8(255*mat2gray(R2{1,4}))})
title('Laplacian Pyramid for Input 2(DCP) (level 1-->level 4)')  

R3 = laplacian_pyramid(img3(:, :, 1), level);
G3 = laplacian_pyramid(img3(:, :, 2), level);
B3 = laplacian_pyramid(img3(:, :, 3), level);
figure,montage({uint8(255*mat2gray(R3{1,1})),uint8(255*mat2gray(R3{1,2})),uint8(255*mat2gray(R3{1,3})),uint8(255*mat2gray(R3{1,4}))})
title('Laplacian Pyramid for Input 3(DCP) (level 1-->level 4)')  

    
% Multiscale fusion
 for i = 1 : level
     R_r{i} = Weight1{i} .* R1{i} + Weight2{i} .* R2{i} + Weight3{i} .* R3{i};
     R_g{i} = Weight1{i} .* G1{i} + Weight2{i} .* G2{i} + Weight3{i} .* G3{i};
     R_b{i} = Weight1{i} .* B1{i} + Weight2{i} .* B2{i} + Weight3{i} .* B3{i};
 end
 
% reconstruction
R = pyramid_reconstruct(R_r);
G = pyramid_reconstruct(R_g);
B = pyramid_reconstruct(R_b);

fusion = cat(3, uint8(255*mat2gray(R)), uint8(255*mat2gray(G)), uint8(255*mat2gray(B)));
figure
imshow(fusion)
title('FUSION');

entro3=entropy(fusion);
fprintf('The entropy value for FUSED OUTPUT is %f\n',entro3);
MSE3 = immse(fusion,Ref_Image);
fprintf('The MSE value for FUSED OUTPUT is %0.4f\n',MSE3);
PSNR3 = psnr(fusion,Ref_Image);
fprintf('The PSNR value for FUSED OUTPUT is %0.4f\n', PSNR3);
SSIM3 = ssim(fusion,Ref_Image);
fprintf('The SSIM value for FUSED OUTPUT is %0.4f\n',SSIM3);
uicm3= UICM(fusion);
fprintf('The UICM value for FUSED OUTPUT is %f\n',uicm3);
uism3=UISM(fusion);
fprintf('The UISM value for FUSED OUTPUT is %f\n',uism3);
uiconm3=UIConM(fusion);
fprintf('The UIConM value for FUSED OUTPUT is %f\n',uiconm3);
uiqm3 = UIQM(fusion);
fprintf('The UIQM value for FUSED OUTPUT is %f\n',uiqm3);
uciqe3=UCIQE(fusion);
fprintf('The UCIQE value for FUSED OUTPUT is %f\n\n',uciqe3);



T = [MSE PSNR SSIM uiqm uicm uism uiconm uciqe MSE1 PSNR1 SSIM1 uiqm1 uicm1 uism1 uiconm1 uciqe1 MSE2 PSNR2 SSIM2 uiqm2 uicm2 uism2 uiconm2 uciqe2 MSE3 PSNR3 SSIM3 uiqm3 uicm3 uism3 uiconm3 uciqe3];


% filename = 'set2.xlsx';
% 
% xlswrite(filename, {entro,MSE,PSNR,SSIM,uicm,uism,uiconm,uiqm,uciqe,entro1,MSE1,PSNR1,SSIM1,uicm1,uism1,uiconm1,uiqm1,uciqe1,entro2,MSE2,PSNR2,SSIM2,uicm2,uism2,uiconm2,uiqm2,uciqe2,entro3,MSE3,PSNR3,SSIM3,uicm3,uism3,uiconm3,uiqm3,uciqe3}, 'Sheet1', ['A' num2str(x)]);




% ds=spreadsheetDatastore('sheet.xlsx',k);
% ds.Sheets = '';
% allSheets=read(ds);

% imwrite(white_balance,'results_base\INPUT1(WB).jpg');
% imwrite(RGB_Image_o,'results_base\Original_image.jpg');
% imwrite(Ref_Image,'results_base\Reference_image.jpg');
% imwrite(dehazed_image,'results_base\INPUT2(DCP).jpg');
% 
% imwrite(uint8(255*mat2gray(WG1)),'results_base\WG1.jpg');
% imwrite(uint8(255*mat2gray(WL1)),'results_base\WL1.jpg');
% imwrite(uint8(255*mat2gray(WS1)),'results_base\WS1.jpg');
% imwrite(uint8(255*mat2gray(WE1)),'results_base\WE1.jpg');
% 
% imwrite(uint8(255*mat2gray(WG2)),'results_base\WG2.jpg');
% imwrite(uint8(255*mat2gray(WL2)),'results_base\WL2.jpg');
% imwrite(uint8(255*mat2gray(WS2)),'results_base\WS2.jpg');
% imwrite(uint8(255*mat2gray(WE2)),'results_base\WE2.jpg');
% 
% imwrite(uint8(255*mat2gray(W1)),'results_base\NormalisedWeight_1.jpg');
% imwrite(uint8(255*mat2gray(W2)),'results_base\NormalisedWeight_2.jpg');
% 
% 
% for i=1:5
%     imwrite(uint8(255*mat2gray(Weight1{1,i})), fullfile('results_base\Gaussian_pyramids_of_NormalisedWeight_1', strcat(strcat('G_', num2str(i)),'.jpg')));
% end
% for i=1:5
%     imwrite(uint8(255*mat2gray(Weight2{1,i})), fullfile('results_base\Gaussian_pyramids_of_NormalisedWeight_2', strcat(strcat('G_', num2str(i)),'.jpg')));
% end
% 
% 
% for i=1:5
%     imwrite(uint8(255*mat2gray(R1{1,i})), fullfile('results_base\Laplacian_pyramids_of_INPUT1', strcat(strcat('L_', num2str(i)),'.jpg')));
% end
% for i=1:5
%     imwrite(uint8(255*mat2gray(R2{1,i})), fullfile('results_base\Laplacian_pyramids_of_INPUT2', strcat(strcat('L_', num2str(i)),'.jpg')));
% end
% 
% imwrite(fusion,'results_base\FUSED_OUTPUT.jpg');






