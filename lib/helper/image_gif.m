clear all
close all

% original images file path and outfile name and image number.
datapath = '/Users/liubei/Desktop/test_image_centered_1.0/';
filename = 'center1.gif';
nImages = 30;
steps = 6;

% define pich area and canvas to hold image data.
pick_range_r = [59:430];
pick_range_c = [140:516];
row_size = pick_range_r(end) - pick_range_r(1) + 1;
col_size = pick_range_c(end) - pick_range_c(1) + 1;
canvas = zeros(row_size * 3, col_size * 10, 3);

fig = figure; hold on

% import image data.
im = cell(steps, nImages);
for step = 1:steps
    for idx = 1:nImages
        im{step, idx} = imread([datapath 'im_' num2str(idx) '_step_' num2str(step-1) '.png']);
    end
end

% read image data and put into canvas.
for step = 1:steps
    for im_row = 1:3
        for im_col = 1:10
            canvas(1 + row_size*(im_row-1):row_size + row_size*(im_row-1), 1 + col_size*(im_col-1):col_size + col_size*(im_col-1), :) = im{step,(im_row-1)*10+im_col}(pick_range_r, pick_range_c, :);
        end
    end
    imshow(canvas);
    set(gca,'position',[0 0 1 1],'units','normalized')
    axis equal; axis off;colormap gray
    frame = getframe(fig);
    img = frame2im(frame);
    [A,map] = rgb2ind(img,256);

    % if first image to create gif file, if not append to existed file.
    if step == 1
        imwrite(A,map,filename,'gif', 'Loopcount',inf);
    else
        imwrite(A,map,filename,'gif','WriteMode','append');
    end
end
