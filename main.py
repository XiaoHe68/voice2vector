import argparse
import os
import feature_extract
import data_info
from implement import Compress2OneFiel,feat_x,get_feat,save_feat


if __name__=='__main__':


    devlst=['cvector','evector','rvector']
    direclst=['mfcc','mfcc_delta','mfcc_delta_delta','logfbank','fbank','hdcc','idct','spectrum_power']
    featlst=devlst+direclst

    parser = argparse.ArgumentParser(description='使用说明')  # 首先创建一个ArgumentParser对象
    parser.add_argument('-dev',action='store_true', help='标识 develop 阶段')
    parser.add_argument('-file', action='store_true', help='单文件提取模式')
    parser.add_argument('-dir', action='store_true', help='文件夹提取模式，提取文件夹下所有文件，需要自己编辑data_info文件中的方法类')
    parser.add_argument('-vad', action='store_true', help='是否对 wav 文件进行vad处理')

    parser.add_argument('--savefmt',choices=['txt','npy'],help='单文件模式下特征保存格式')
    parser.add_argument('--infile', help='输入文件')
    parser.add_argument('--outfile', help='输出文件')
    parser.add_argument('--kpath', help='x-vector develop 阶段产生的路径文件（可选）')

    parser.add_argument('--outfolder',help='输出文件路径，默认值为data_info中定义的值（可选）')
    parser.add_argument('--wavfolder', help='输入文件路径，默认值为data_info中定义的值（可选）')


    parser.add_argument('--winlen',type=int, help='窗口长度，单位毫秒',default=25)
    parser.add_argument('--winstep',type=int, help='窗口移动步长，单位毫秒',default=5)
    parser.add_argument('--ndim',type=int, help='目标特征维度')
    parser.add_argument('--dbclass',help='data_info 中的数据库类名')
    parser.add_argument('--idctfile', help='develop 阶段产生的idct系数文件完整路径(可选)')

    parser.add_argument('--feat',
                        choices=featlst,
                        required=True,
                        help='可选特征类型')

    args = parser.parse_args()
    # 定义提取方法
    fe = feature_extract.voice_feature()
    fe.set_feat_type(args.feat)
    fe.p_win_len=args.winlen*0.001
    fe.p_win_step=args.winstep*0.001

    fe.p_vad=args.vad

    db=None

    if args.dir or args.dev:
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

        if args.idctfile:
            db.idctcoefffile=args.idctfile
        else:
            db.idctcoefffile = db.outfolder + '/idct.npy'


    if args.feat in devlst:
        if args.dev:
            if not os.path.isfile(db.idctcoefffile):
                print('idct file not exit,creat idec file...')
                fe.set_feat_type('idct')
                Compress2OneFiel(fe,db,ndim=None)
            feat_x(db,args.feat,[])
        else:
            if args.file or args.dir:
                if args.dir:
                    kpath = db.outfolder + '/' + args.feat + '_kpath_1.txt'
                elif args.kpath:
                    kpath = args.kpath
                else:
                    kpath=None

                if not os.path.isfile(kpath):
                    print(kpath)
                    raise FileExistsError('kpath 文件不存在请先进行开发过程')

                fe.set_feat_type('xvector')
                if args.file:
                    if args.infile is None:
                        raise ValueError('请增加键值对指向输入文件：--infile=')
                    if args.outfile is None:
                        raise ValueError('请增加键值对指向输出文件：--outfile=')
                    savefmt=args.savefmt if args.savefmt else 'txt'

                    if args.ndim and args.ndim > 0:
                        save_feat(get_feat(fe, args.infile, kpath=kpath, ndim=args.ndim), args.outfile, savefmt)
                    else:
                        save_feat(get_feat(fe, args.infile, kpath=kpath), args.outfile, savefmt)
                else:
                    if args.ndim and args.ndim > 0:
                        Compress2OneFiel(fe, db,ndim=args.ndim,kpath=kpath)
                    else:
                        Compress2OneFiel(fe, db,kpath=kpath)
            else:
                print('什么都没做！请设置-dir 或 -file')
            print('finish! ')


    if args.feat in direclst:
        if args.file or args.dir:
            if args.dir:
                if args.ndim and args.ndim > 0:
                    Compress2OneFiel(fe, db,ndim=args.ndim)
                else:
                    Compress2OneFiel(fe, db)
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
