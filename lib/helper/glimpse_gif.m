clear all
close all

% original images file path and outfile name and image number.
datapath = '/Users/liubei/Desktop/test_image_centered_1.0/';
nImages = 30;
steps = 6;
glimpse_scale_num = 1;
row_size = 9;  % modify according to image size = image_size + 1.
col_size = 9;

% define pich area and canvas to hold image data.
canvas = ones(row_size * 3, col_size * 10);


im = cell(steps, nImages);

for scale_id = 1:glimpse_scale_num
    fig = figure; hold on
    % import image data.
    for step = 1:steps
        for idx = 1:nImages
            im{step, idx} = imread([datapath 'im_' num2str(idx) '_glimpse_' num2str(scale_id-1) '_step_' num2str(step-1) '.png']);
        end
    end

    % read image data and put into canvas.
    for step = 1:steps
        for im_row = 1:3
            for im_col = 1:10
                canvas(2 + row_size*(im_row-1):row_size + row_size*(im_row-1), 2 + col_size*(im_col-1):col_size + col_size*(im_col-1)) = im{step,(im_row-1)*10+im_col};
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
            imwrite(A,map,['glimpse_scale_' num2str(scale_id) '.gif'],'gif', 'Loopcount',inf);
        else
            imwrite(A,map,['glimpse_scale_' num2str(scale_id) '.gif'],'gif','WriteMode','append');
        end
    end
end
