import argparse
import os
import feature_extract
import data_info
from implement import Compress2OneFiel,feat_xvector,get_feat,get_feat_xvector,save_feat


if __name__=='__main__':


    devlst=['cvector','evector','rvector']
    direclst=['mfcc','mfcc_delta','mfcc_delta_delta','logfbank','fbank','hdcc','idct','spectrum_power','hdcc_fs']
    featlst=devlst+direclst+['v2v']

    parser = argparse.ArgumentParser(description='usage ')
    parser.add_argument('-file', action='store_true', help='Single file extraction mode')
    parser.add_argument('-dir', action='store_true', help='Folder extract mode, extract all files under the folder,'
                                                          ' you need to edit the method class in the data_info.py '
                                                          'file yourself')


    parser.add_argument('--savefmt',choices=['txt','npy'],help='Feature saving format in single file mode')
    parser.add_argument('--infile', help='Input file path')
    parser.add_argument('--outfile', help='Output file path')

    parser.add_argument('--outfolder',help='Output file path with default value defined in data_info (optional)')
    parser.add_argument('--wavfolder', help='Enter the file path, which defaults to the value defined in data_info (optional)')

    parser.add_argument('-vad', action='store_true', help='Whether to vad wav files')
    parser.add_argument('--winlen',type=int, help='Window length in milliseconds',default=25)
    parser.add_argument('--winstep',type=int, help='Window move step in milliseconds',default=5)
    parser.add_argument('--ndim',type=int, help='Objective feature dimension')


    parser.add_argument('--dbclass',help='The name of the database class in data_info.py')
    parser.add_argument('--pre_data', help='rvector、evector、cvector的输入文件')
    parser.add_argument('--pnumber', type=int,help='rvector、evector、cvector时候选择聚类人数')

    parser.add_argument('--featPath', help='v2v must other feat optional')
    parser.add_argument('--saveName', help='v2v must other feat optional')

    parser.add_argument('--feat',
                        choices=featlst,
                        required=True,
                        help='Optional feature types')

    args = parser.parse_args()
    # 定义提取方法
    fe = feature_extract.voice_feature()
    fe.set_feat_type(args.feat)
    fe.p_win_len=args.winlen*0.001
    fe.p_win_step=args.winstep*0.001
    fe.p_vad=args.vad
    db=None



    if args.dir:

        if args.dbclass is None:
            raise ValueError('请增加键值对：--dbclass=')

        # 定义数据来源
        dbmolde = getattr(data_info, args.dbclass, None)
        if dbmolde is None:
            raise ModuleNotFoundError('指定数据库类 %s 不存在于文件 data_info.py 中，请修改后重试！'% args.dbclass)
        db=dbmolde()

        if args.outfolder:
            db.outfolder=args.outfolder
        else:
            db.outfolder += '/' + str(int(fe.p_win_len * 1000)) + 'ms'

        if args.wavfolder:
            db.wavefolder = args.wavfolder


    if args.feat in devlst:

        if args.ndim and args.ndim>0:
            pass
        else:
            raise ValueError('--ndim 参数不正确')

        if args.file:
            if args.infile is None:
                raise ValueError('请增加键值对指向输入文件：--infile=')
            if args.outfile is None:
                raise ValueError('请增加键值对指向输出文件：--outfile=')
            get_feat_xvector(fe, args.infile, ndim=args.ndim,x=args.feat,outputfile=args.outfile)
        else:
            if not os.path.isfile(args.pre_data):
                raise FileNotFoundError('--pre_data 参数设置不正确')
            if not args.outfolder:
                raise FileNotFoundError('--outfolder 参数设置不正确')


            if args.pnumber and args.pnumber>0:
                feat_xvector(args.pre_data,args.outfolder,args.feat,[args.ndim],args.pnumber)
            else:
                raise ValueError('--pnumber 参数不正确')
        print('finish! ')


    if args.feat in direclst:
        if args.file or args.dir:
            if args.dir:
                if args.ndim and args.ndim > 0:
                    Compress2OneFiel(fe, db,args.feat,ndim=args.ndim)
                else:
                    Compress2OneFiel(fe, db,args.feat)
            else:
                if args.infile is None:
                    raise ValueError('请增加键值对指向输入文件：--infile=')
                if args.outfile is None:
                    raise ValueError('请增加键值对指向输出文件：--outfile=')
                savefmt = args.savefmt if args.savefmt else 'txt'
                if args.ndim and args.ndim > 0:
                    save_feat(get_feat(fe, args.infile,ndim=args.ndim), args.outfile, savefmt)
                else:
                    save_feat(get_feat(fe, args.infile), args.outfile, savefmt)
        else:
            print('什么都没做！请设置-dir 或 -file')

    if args.feat in ['v2v']:
        if args.featPath:
            if not os.path.isfile(args.featPath):
                print(args.featPath)
                raise FileNotFoundError('kpath 文件不存在请先进行开发过程')

            if args.kpath:
                kpath = args.kpath
            else:
                kpath = ''
            if not os.path.isfile(kpath):
                print(kpath)
                raise FileNotFoundError('kpath 文件不存在请先进行开发过程')

            outfolder=None
            if args.outfolder:
                outfolder = args.outfolder
            if not os.path.exists(outfolder):
                raise FileNotFoundError('输出路径不存在！')

            if args.ndim and int(args.ndim)>0:
                ndim=int(args.ndim)
            else:
                raise ValueError('--ndim 必须为大于0的整数')

            if args.saveName is None:
                raise ValueError('--saveName 必须赋值')

            import numpy as np
            import columu_cluster
            feat1=np.load(args.featPath)
            feat= columu_cluster._merge_feat(feat1, kpath, ndim)
            print('saving feat file...')
            save_feat(feat,outfolder + '/' +args.saveName ,'npy')
            print('feat shape:', feat.shape)
        else:
            raise FileNotFoundError('--featPath 不能为空值')



