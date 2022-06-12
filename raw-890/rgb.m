function imind = rgb(img)
  
  % replace `PATH_TO_GRAY_IMAGE` with actual image file
%   img = imread('C:\Users\vasu\Desktop\matcodes\57.jpg');
  
  
  % convert grayscale image to indexed image
  imind=gray2ind(img);
  
  % plot the results
  subplot(1,2,1),imshow(img);
  title('Original RGB image','FontSize',18);
  
  subplot(1,2,2),imshow(imind);
  title('Indexed image','FontSize',18);
  
end