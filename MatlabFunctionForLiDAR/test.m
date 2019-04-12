DataRoot = 'C:\Users\dxjfo\Desktop\M_H_m22_3_164\';
IS_SHOW=1;
OutputRoot='C:\Users\dxjfo\Desktop\output\';

for nFram=1:1:137
    pcData = HDLAnalyserNew(DataRoot,OutputRoot,nFram,IS_SHOW);
end