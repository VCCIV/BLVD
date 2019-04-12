%% convert smaller binary file .txt(~298kB) to 3D point cloud, this function costs about 60~90ms.
function pcData = HDLAnalyserNew(DataRoot,OutputRoot,nFrm,IS_SHOW)

clc; close all;
str = sprintf('%d.txt', nFrm);
DataDir = fullfile(DataRoot, str);

Ang = 1.0;
R = [cosd(Ang) sind(Ang) 0; -sind(Ang) cosd(Ang) 0; 0 0 1];
T = [1.1; 0.0; 0.0];

%% Correction parameter rectified.
ParaDir = 'HDLS3Calib.txt';
% id minIntensity maxIntensity, rotCorrection_, vertCorrection_,distCorrection_,distCorrectionX_,distCorrectionY_,
% vertOffsetCorrection_,horizOffsetCorrection_,focalDistance_,focalSlope_
tmp = importdata(ParaDir, ' ', 1);
tmp = tmp.data;
SeqMap = [36 37 58 59  38 39 32 33 40 41 34 35 48 49 42 43 50 51 44 45 52 53 46 47 60 61 54 55 62 63 56 57 4  5  26 27 6  7  0  1  8 9 2  3  16 17 10 11 18 19 12 13 20 21 14 15 28 29 22 23 30 31 24 25];
% SeqMap = 1:1:64;
[~, Idx] = sort(SeqMap);
Para = tmp( Idx,: );
CosSinTab = [cosd( (0:1:3999) * 0.09 );
    sind( (0:1:3999) * 0.09) ];
%% Read binary data from .txt.
fid = fopen(DataDir, 'rb' );
BinaryData = fread(fid, 'uint8=>uint16');
UDPNum = length(BinaryData) / 1206;
A = reshape( BinaryData, 1206, UDPNum);
fclose(fid);
%% Extract raw angle, distance and intensity.
RawDistData = zeros(3, UDPNum * 6, 64);
for id = 1 : 2 : 6
    Idx = (1 + (id-1)*200) : 1 : id * 200;
    Buffer = A(Idx, :);
    
    UpLow1 = Buffer(1, :) + 256 * Buffer(2, :);
    UpLow2 = Buffer(101, :) + 256 * Buffer(102, :);
    Idx = find(UpLow1 ~= hex2dec('eeff') & UpLow2 ~= hex2dec('ddff') );
    if ~isempty(Idx)
        disp('The Up block(0xEEFF) and the Low Block(0xDDFF) is not right!\n');
    end
    RotAng1 = Buffer(3, :) + 256 * Buffer(4, :);
    RotAng2 = Buffer(103, :) + 256 * Buffer(104, :);
    Idx = find( (RotAng1 == RotAng2) ~= 1 );
    if ~isempty(Idx)
        disp('Two Angle in the Block is not equal! \n');
    end
    % AngID = floor(RotAng1 / 9);
    Idx = ( 1 + (id-1) * UDPNum ) : 1 : id * UDPNum;
    for i = 0 : 1 : 31
        Id0 = SeqMap(i+1)+1;
        Id1 = SeqMap(i+33)+1;
        RawDistData(1, Idx, Id0)  = RotAng1; % AngID;
        RawDistData(2, Idx, Id0)  = Buffer(i*3+5, :) + 256 * Buffer(i*3+6, :);
        RawDistData(3, Idx, Id0)  = Buffer(i*3+7, :);
        RawDistData(1, Idx, Id1) = RotAng1; % AngID;
        RawDistData(2, Idx, Id1) = Buffer(i*3+105, :) + 256 * Buffer(i*3+106, :);
        RawDistData(3, Idx, Id1) = Buffer(i*3+107, :);
    end
end
%% median filter. it seems this will introduce some noise.....
USE_MEDIAN_FILTER = 0;
if USE_MEDIAN_FILTER
    for i = 1 : 1 : 64
        Dist = RawDistData(2, :, i);
        % RawDistData(2, :, i) = medfilt1( Dist, n );
        tmp = [ Dist( (end-n+1):1:end) Dist Dist(1:1:n) ];
        tmp = medfilt1( tmp, n );
        RawDistData(2, :, i) = tmp((n+1):1:end-3);
    end
end
n = 3;

%% analysis data.
pcData = [];
Radius = [];
RotAngID = [];
RotArray = []; 
RadArray = []; 
for i = 1 : 1 : 64
    if i == 37
        bTest = 1;
    end
    UDPNum_Indicator = repmat(1:1:UDPNum, 1, 6);
    if i == 37
        bTest = 1; 
    end
    RotAngID = RawDistData(1, :, i);
    Radius   = RawDistData(2, :, i);
    Intensity = RawDistData(3, :, i);
    Idx0 = find(Radius > 0.0 );
    UDPNum_Indicator = UDPNum_Indicator(Idx0);
    RotAngID = RotAngID(Idx0);
    Radius   = Radius(Idx0);
    Intensity = Intensity(Idx0);
    
    if i == 54
        myIdx = find(RotAngID == 8478 & UDPNum_Indicator == 26);
        bTest = 1; 
    end
    
    % id minIntensity maxIntensity, rotCorrection_, vertCorrection_,distCorrection_,distCorrectionX_,distCorrectionY_,
    % vertOffsetCorrection_,horizOffsetCorrection_,focalDistance_,focalSlope_
    Gama = Para(i, 4);  % rotational angle.
    Beta = Para(i, 5);  % vertical angle.
    cosVertAngle = cosd(Beta);
    sinVertAngle = sind(Beta);
    D = Para(i, 6);
    DistCorrX = Para(i, 7);
    DistCorrY = Para(i, 8);
    vOffsetCorr = Para(i, 9) / 100.0;
    hOffsetCorr = Para(i, 10) / 100.0;
    distancel = 0.2*Radius;
    distance = distancel+D;
    distance = distance / 100;
    cosVertAngle = cosd(Beta);
    sinVertAngle = sind(Beta);
    AngID = floor(RotAngID/9); 
    RotAng = AngID * 0.09 - Gama;   % 0.09 is the angle's resolution.
    
    c = CosSinTab(1, AngID+1 );
    s = CosSinTab(2, AngID+1 );
    cosRotAngle = c * cosd(Gama) + s * sind(Gama);
    sinRotAngle = s * cosd(Gama) - c * sind(Gama);
    xyDistance = distance * cosVertAngle;
    xx =  xyDistance .* sinRotAngle;
    idx = find(xx < 0);
    xx(idx) = -xx(idx);
    yy = xyDistance .* cosRotAngle;
    idx = find(yy<0);
    yy(idx) = -yy(idx);
    distanceCorrX = DistCorrX*(xx-240)/(2504-240)+DistCorrX;
    distanceCorrY = DistCorrY*(yy-193)/(2504-193)+DistCorrY;
    idx = find(distancel>2500);
    distanceCorrX(idx) = D;
    distanceCorrY(idx) = distanceCorrX(idx);
    
    distancel = distancel / 100;
    distanceCorrX = distanceCorrX / 100;
    distanceCorrY = distanceCorrY / 100;
    
    distance = distancel + distanceCorrX;
    xyDistance = distance .* cosVertAngle;
    x =  xyDistance .* cosRotAngle + hOffsetCorr .* sinRotAngle;
    
    distance = distancel + distanceCorrY;
    xyDistance = distance .* cosVertAngle;
    y = -xyDistance .* sinRotAngle + hOffsetCorr .* cosRotAngle;
    z = distance * sinVertAngle + vOffsetCorr * cosVertAngle;
    tmp = [x; y; z; Intensity; i * ones(1, length(x)); UDPNum_Indicator];
    Idx = find( xyDistance >= 2.0 & xyDistance <= 100.0 );
    pcData = [pcData tmp(:, Idx)];
    RotArray = [RotArray RotAngID(Idx)]; 
    RadArray = [RadArray Radius(Idx)]; 
end
pcData(1:3, :) = R * pcData(1:3, : ) + repmat(T, 1, size(pcData, 2) );
pcData(4, :)=pcData(4, :)/255;
% A = [pcData(5, :); RotArray; pcData(6, :); RadArray; pcData(1:3, :)]'; 
% save('C:\Program Files\MATLAB\R2015a\bin\data_tracking\AInfo.mat', 'A'); 
result_str = sprintf('%06d.bin', nFrm);
output_file_dir=fullfile(OutputRoot,result_str);
fid = fopen(output_file_dir,"wb"); 
fwrite(fid,pcData(1:4, :),"double"); 
fclose(fid);


if IS_SHOW
    figure;
    hold on;
    axis equal;
    plot3(pcData(1, :), pcData(2, :), pcData(3, :), 'b.' );
    figure;
    hold on;
    view(3);
    showPointCloud(pcData(1:3, :)');
    Idx = find(pcData(end-1, :) == 1);
    showPointCloud(pcData(1:3, Idx)', 'r');
    Idx = find(pcData(end, :) == 1);
    showPointCloud(pcData(1:3, Idx)', 'k');
    pause(1);  
end
end



