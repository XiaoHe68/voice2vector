# voice2vector 
voice2vector是一个对说话者语音进行特征提取的工具库可以用来提取以下声纹特征：
cvector,evector,rvector,mfcc,mfcc_delta,mfcc_delta_delta,logfbank,fbank,hdcc,idct,spectrum_power

本程序是基于 Python3 完成
部分功能依赖库如下：  
python_speech_features==0.6  
scipy>=1.1.0  
numpy>=1.16.0  
sphfile>=1.0.1  


使用帮助：

```bash
$ python3 main.py -h
使用说明

optional arguments:  
  -h, --help            show this help message and exit  
  -dev                  标识 develop 阶段  
  -file                 单文件提取模式  
  -dir                  文件夹提取模式，提取文件夹下所有文件，需要自己编辑data_info文件中的方法类  
  -vad                  是否对 wav 文件进行vad处理  
  --savefmt {txt,npy}   单文件模式下特征保存格式  
  --infile INFILE       输入文件  
  --outfile OUTFILE     输出文件  
  --kpath KPATH         x-vector develop 阶段产生的路径文件（可选）  
  --outfolder OUTFOLDER
                        输出文件路径，默认值为data_info中定义的值（可选）  
  --wavfolder WAVFOLDER
                        输入文件路径，默认值为data_info中对定义的值（可选）  
  --winlen WINLEN       窗口长度，单位毫秒  
  --winstep WINSTEP     窗口移动步长，单位毫秒  
  --ndim NDIM           目标特征维度  
  --dbclass DBCLASS     data_info 中的数据库类名  
  --idctfile IDCTFILE   develop 阶段产生的idct系数文件完整路径(可选)  
  --feat {cvector,evector,rvector,mfcc,mfcc_delta,mfcc_delta_delta,logfbank,fbank,hdcc,idct,spectrum_power}
                        可选特征类型  
```
feat中的cvector,evector,rvector分别是对应层次聚类中使用的不同距离度量方法：余弦相似度、欧式距离、皮尔逊相关系数。
