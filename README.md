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
$ python3 x-vector.py -h
usage

optional arguments:
  -h, --help            show this help message and exit
  -dev                  Identify develop stage
  -file                 Single file extraction mode
  -dir                  Folder extract mode, extract all files under the
                        folder, you need to edit the method class in the
                        data_info.py file yourself
  -vad                  Whether to vad wav files
  --savefmt {txt,npy}   Feature saving format in single file mode
  --infile INFILE       Input file path
  --outfile OUTFILE     Output file path
  --kpath KPATH         X-vector develop path files (optional)
  --outfolder OUTFOLDER
                        Output file path with default value defined in
                        data_info (optional)
  --wavfolder WAVFOLDER
                        Enter the file path, which defaults to the value
                        defined in data_info (optional)
  --winlen WINLEN       Window length in milliseconds
  --winstep WINSTEP     Window move step in milliseconds
  --ndim NDIM           Objective feature dimension
  --dbclass DBCLASS     The name of the database class in data_info.py
  --idctfile IDCTFILE   Complete path to idct coefficient file generated
                        during the develop phase (optional)
  --feat {cvector,evector,rvector,mfcc,mfcc_delta,mfcc_delta_delta,logfbank,fbank,hdcc,idct,spectrum_power,hdcc_fs}
                        Optional feature types
```
feat中的cvector,evector,rvector分别是对应层次聚类中使用的不同距离度量方法：余弦相似度、欧式距离、皮尔逊相关系数。

# usage
Please modify the `data_info.py` file first,the variable `wavefolder` is where the sound file is located,`outfolder` is where the feat file output folder,`extension` leave the default value `.wav` if you don't understand what that means.  
the function `generatefunc` is used to create unique identifiers for sentences and which sentences someone says.
```bash
x-vector(Cosine similarity) :
$ python3 x-vector.py --dbclass=timit_info -dir --winlen=25  --winstep=10 -dev --feat=cvector --ndim=15
$ python3 x-vector.py --dbclass=timit_info -dir --winlen=25  --winstep=10 --feat=cvector --ndim=15

x-vector(Euclidean distance) :
$ python3 x-vector.py --dbclass=timit_info -dir --winlen=25  --winstep=10 -dev --feat=evector --ndim=15
$ python3 x-vector.py --dbclass=timit_info -dir --winlen=25  --winstep=10 --feat=evector --ndim=15

x-vector(Pearson correlation coefficient) :
$ python3 x-vector.py --dbclass=timit_info -dir --winlen=25  --winstep=10 -dev --feat=rvector --ndim=15
$ python3 x-vector.py --dbclass=timit_info -dir --winlen=25  --winstep=10 --feat=rvector --ndim=15

mfcc:
$ python3 x-vector.py --dbclass=timit_info -dir --winlen=25  --winstep=10 --feat=mfcc

use vad：
If you want to use Voice Activity Detection(vad) add the parameter -vad After the command line，example
$ python3 x-vector.py --dbclass=timit_info -dir --winlen=25  --winstep=10 --feat=mfcc -vad
```
If you want to extract the features of a <b>single sentence</b>:
```bash
$ python3 x-vector.py -file --winlen=25 --winstep=10 --feat=mfcc --infile=INPUT_FILEPATH --outfile=OUTPUT_FILEPATH --savefmt=npy
```
