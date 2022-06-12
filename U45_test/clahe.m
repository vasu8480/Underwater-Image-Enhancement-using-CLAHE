function balance=clahe(data)
%% First pass: Count equal pixels
%  data=imread('C:\Users\vasu\Downloads\Quotefancy-326506-3840x2160.jpg');                 % filename : 'london.jpg'

data3=data(:, :, 1);
t=1;                                   % start index of the window (row)
limit=8;                               % window size of the contextual area
endt=limit;                            % end index of the window (row)
eqdata=zeros(size(data3,1),size(data3,2));
for x=1:size(data3,1)
    q=1;                                % start index of the window (column)
    endq=limit;                         % end index of the window (column)
    %% TO move Window to right and bottom, after exceeding the limit 
        for y=1:size(data3,2)
        eqdata(x,y)=0;
        if (x>t+limit-1)
            t=t+limit;
            endt=limit+t-1;
        end
        if (y>q+limit-1)
            q=q+limit;
            endq=limit+q-1;
        end
        if (endt>size(data3,1))
            % t=t-64;
            endt=size(data3,1);
        end
        if (endq>size(data3,2))
            %  q=q-64;
            endq=size(data3,2);
        end
    %% Counting the number of pixels in each contextual area    
        for i=t:endt
            for j=q:endq
                
                if data3(x,y)==data3(i,j)
                    eqdata(x,y)=eqdata(x,y)+1;
                end
                
            end
        end
        
        
    end
end

%% Second Pass: Calculate partial rank, redistributed area and output values.

output=zeros(size(data3,1),size(data3,2));
cliplimit=0.3;                                  % Cliplimit can v6   ary between 0 to 1.
t=1;
endt=limit;
for x=1:size(data3,1)
    q=1;
    endq=limit;
    %% TO move Window to right and bottom, after exceeding the limit
    for y=1:size(data3,2)
        
        cliptotal=0;
        partialrank=0;
        if (x>t+limit-1)
            t=t+limit;
            endt=limit+t-1;
        end
        if (y>q+limit-1)
            q=q+limit;
            endq=limit+q-1;
        end
        if (endt>size(data3,1))
            % t=t-64;
            endt=size(data3,1);
        end
        if (endq>size(data3,2))
            % q=q-64;
            endq=size(data3,2);
        end
        
        %% For each pixel (x,y), compare with cliplimit and accordingly do the clipping. Calculate partialrank. 
        for i=t:endt
            for j=q:endq
                
                
                if eqdata(i,j)>cliplimit
                    
                    incr=cliplimit/eqdata(i,j);
                else
                    incr=1;
                    
                end
                cliptotal=cliptotal+(1-incr);
                
                if data3(x,y)>data3(i,j)
                    partialrank=partialrank+incr;
                    
                end
                
            end
            
        end
        %% New distributed pixel values can be found from redistr and will be incremented by partial rank.
        
        redistr=(cliptotal/(limit*limit)).*data3(x,y);
        output(x,y)=partialrank+redistr;
        
    end
end

r=uint8(output);  

% 2222222222222222222222222222222222222222222
data1=data(:, :, 2);
t=1;                                   % start index of the window (row)
limit=8;                               % window size of the contextual area
endt=limit;                            % end index of the window (row)
eqdata=zeros(size(data1,1),size(data1,2));
for x=1:size(data1,1)
    q=1;                                % start index of the window (column)
    endq=limit;                         % end index of the window (column)
    %% TO move Window to right and bottom, after exceeding the limit 
        for y=1:size(data1,2)
        eqdata(x,y)=0;
        if (x>t+limit-1)
            t=t+limit;
            endt=limit+t-1;
        end
        if (y>q+limit-1)
            q=q+limit;
            endq=limit+q-1;
        end
        if (endt>size(data1,1))
            % t=t-64;
            endt=size(data1,1);
        end
        if (endq>size(data1,2))
            %  q=q-64;
            endq=size(data1,2);
        end
    %% Counting the number of pixels in each contextual area    
        for i=t:endt
            for j=q:endq
                
                if data1(x,y)==data1(i,j)
                    eqdata(x,y)=eqdata(x,y)+1;
                end
                
            end
        end
        
        
    end
end

%% Second Pass: Calculate partial rank, redistributed area and output values.

output1=zeros(size(data1,1),size(data1,2));
cliplimit=0.3;                                  % Cliplimit can v6   ary between 0 to 1.
t=1;
endt=limit;
for x=1:size(data1,1)
    q=1;
    endq=limit;
    %% TO move Window to right and bottom, after exceeding the limit
    for y=1:size(data1,2)
        
        cliptotal=0;
        partialrank=0;
        if (x>t+limit-1)
            t=t+limit;
            endt=limit+t-1;
        end
        if (y>q+limit-1)
            q=q+limit;
            endq=limit+q-1;
        end
        if (endt>size(data1,1))
            % t=t-64;
            endt=size(data1,1);
        end
        if (endq>size(data1,2))
            % q=q-64;
            endq=size(data1,2);
        end
        
        %% For each pixel (x,y), compare with cliplimit and accordingly do the clipping. Calculate partialrank. 
        for i=t:endt
            for j=q:endq
                
                
                if eqdata(i,j)>cliplimit
                    
                    incr=cliplimit/eqdata(i,j);
                else
                    incr=1;
                    
                end
                cliptotal=cliptotal+(1-incr);
                
                if data1(x,y)>data1(i,j)
                    partialrank=partialrank+incr;
                    
                end
                
            end
            
        end
        %% New distributed pixel values can be found from redistr and will be incremented by partial rank.
        
        redistr=(cliptotal/(limit*limit)).*data1(x,y);
        output1(x,y)=partialrank+redistr;
        
    end
end
g=uint8(output1); 
% 333333333333333333333333333333333333333333333333333333333333333333333
data2=data(:, :, 3);
t=1;                                   % start index of the window (row)
limit=8;                               % window size of the contextual area
endt=limit;                            % end index of the window (row)
eqdata=zeros(size(data2,1),size(data2,2));
for x=1:size(data2,1)
    q=1;                                % start index of the window (column)
    endq=limit;                         % end index of the window (column)
    %% TO move Window to right and bottom, after exceeding the limit 
        for y=1:size(data2,2)
        eqdata(x,y)=0;
        if (x>t+limit-1)
            t=t+limit;
            endt=limit+t-1;
        end
        if (y>q+limit-1)
            q=q+limit;
            endq=limit+q-1;
        end
        if (endt>size(data2,1))
            % t=t-64;
            endt=size(data2,1);
        end
        if (endq>size(data2,2))
            %  q=q-64;
            endq=size(data2,2);
        end
    %% Counting the number of pixels in each contextual area    
        for i=t:endt
            for j=q:endq
                
                if data2(x,y)==data2(i,j)
                    eqdata(x,y)=eqdata(x,y)+1;
                end
                
            end
        end
        
        
    end
end

%% Second Pass: Calculate partial rank, redistributed area and output values.

output3=zeros(size(data2,1),size(data2,2));
cliplimit=0.3;                                  % Cliplimit can v6   ary between 0 to 1.
t=1;
endt=limit;
for x=1:size(data2,1)
    q=1;
    endq=limit;
    %% TO move Window to right and bottom, after exceeding the limit
    for y=1:size(data2,2)
        
        cliptotal=0;
        partialrank=0;
        if (x>t+limit-1)
            t=t+limit;
            endt=limit+t-1;
        end
        if (y>q+limit-1)
            q=q+limit;
            endq=limit+q-1;
        end
        if (endt>size(data2,1))
            % t=t-64;
            endt=size(data2,1);
        end
        if (endq>size(data2,2))
            % q=q-64;
            endq=size(data2,2);
        end
        
        %% For each pixel (x,y), compare with cliplimit and accordingly do the clipping. Calculate partialrank. 
        for i=t:endt
            for j=q:endq
                
                
                if eqdata(i,j)>cliplimit
                    
                    incr=cliplimit/eqdata(i,j);
                else
                    incr=1;
                    
                end
                cliptotal=cliptotal+(1-incr);
                
                if data2(x,y)>data2(i,j)
                    partialrank=partialrank+incr;
                    
                end
                
            end
            
        end
        %% New distributed pixel values can be found from redistr and will be incremented by partial rank.
        
        redistr=(cliptotal/(limit*limit)).*data2(x,y);
        output3(x,y)=partialrank+redistr;
        
    end
end
b=uint8(output3);

balan = cat(3,r,g,b);figure;
balance=im2double(balan);
end