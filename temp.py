
# 处理DUBA训练集的路径，去掉前缀 /home/public
# # python
# import os
#
# input_path = r'DAUB/train.txt'
# output_path = r'DAUB/train_stripped.txt'
# prefix = '/home/public'
#
# with open(input_path, 'r', encoding='utf-8') as fin, \
#      open(output_path, 'w', encoding='utf-8') as fout:
#     for line in fin:
#         s = line.rstrip('\n')
#         if not s:
#             fout.write('\n')
#             continue
#         parts = s.split(None, 1)  # 分割为路径和其余部分
#         path = parts[0]
#         rest = parts[1] if len(parts) > 1 else ''
#         if path.startswith(prefix):
#             path = path[len(prefix):] or '/'
#         fout.write(path + ((' ' + rest) if rest else '') + '\n')
#
# print(f"写入完成: {os.path.abspath(output_path)}")


# # 检查 pkl 文件中的路径是否与 annotation 中的路径一致
# import pickle, os

# pkl_file = 'mediate_data/emb_train_ITSDT.pkl'
# data = pickle.load(open(pkl_file,'rb'))

# print('类型：', type(data))
# print('样本数量：', len(data))

# # 看看前几个 key 和对应的 value
# for i,(k,v) in enumerate(data.items()):
#     if i >= 10: break
#     print(i, repr(k), type(v))
#     # 如果 value 是 tuple/list，可以继续展开查看
#     # print('   ->', v)





# # 修改 pkl 文件中的路径，去掉前缀 /home/public
# import pickle, os

# def strip_prefix(d, prefix='/home/public'):
#     newd = {}
#     for k, v in d.items():
#         nk = k
#         # if nk.startswith(prefix):
#         #     nk = nk[len(prefix):]      # 删除前缀
#         # 如果删除后还以 / 开头，去掉它
#         if nk.startswith('/'):
#             nk = nk[1:]
#         newd[nk] = v
#     return newd

# for fname in ('emb_train_DAUB.pkl', 'emb_train_ITSDT.pkl','emb_train_IRDST-H.pkl','motion_relation_DAUB.pkl','motion_relation_ITSDT.pkl','motion_relation_IRDST-H.pkl'):
#     path = os.path.join('mediate_data', fname)
#     if not os.path.isfile(path):
#         continue
#     print('processing', path)
#     with open(path, 'rb') as f:
#         data = pickle.load(f)

#     # # 备份一份原始文件
#     # bak = path + '.bak'
#     # if not os.path.exists(bak):
#     #     os.rename(path, bak)

#     fixed = strip_prefix(data, '/home/public')
#     with open(path, 'wb') as f:
#         pickle.dump(fixed, f)

#     print('  原始 key 示例:', next(iter(data.keys())))
#     print('  修正后 key 示例:', next(iter(fixed.keys())))
#     print('  记录数', len(fixed))



# # 检查代码
# import os
# import numpy as np
# from PIL import Image

# image_size = 512
# num_frame = 5
# def cvtColor(image):
#     if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
#         return image 
#     else:
#         image = image.convert('RGB')
#         return image 
    
# file_name = "DAUB/data5/9.bmp"

# image_dir  = os.path.dirname(file_name)
# basename   = os.path.splitext(os.path.basename(file_name))[0]
# image_id   = int(basename)
# image_data = []
# h, w = image_size, image_size

# for id in range(num_frame):
#     frame_id = max(image_id - id, 0)
#     img_path = os.path.join(image_dir, f"{frame_id}.bmp")
#     if not os.path.isfile(img_path):
#         raise FileNotFoundError(f"frame image not found: {img_path}")
#     img = Image.open(img_path)
#     img = cvtColor(img)
    
# print(image_id)
# print(img_path)

# with Image.open(img_path) as img:
#     img = cvtColor(img)
#     print('图像尺寸:', img.size)





import pickle, os, pprint
import numpy as np

def describe(obj, indent=0, max_depth=2):
    """打印对象的简单描述（递归最多 max_depth 层）"""
    pad = '  ' * indent
    if max_depth < 0:
        print(pad + '...')
        return
    if isinstance(obj, dict):
        print(pad + f'dict, {len(obj)} keys')
        for i,(k,v) in enumerate(obj.items()):
            if i >= 5: 
                print(pad + '  ...')
                break
            print(pad + '  key:', repr(k))
            describe(v, indent+2, max_depth-1)
    elif isinstance(obj, (list, tuple)):
        print(pad + f'{type(obj).__name__}, len={len(obj)}')
        for i,v in enumerate(obj):
            if i >= 3:
                print(pad + '  ...')
                break
            describe(v, indent+1, max_depth-1)
    elif isinstance(obj, np.ndarray):
        print(pad + f'numpy.ndarray, shape={obj.shape}, dtype={obj.dtype}')
    else:
        print(pad + f'{type(obj).__name__}: {repr(obj)[:80]}')

pkl_file = 'mediate_data/motion_relation_DAUB.pkl'
data = pickle.load(open(pkl_file,'rb'))

print('类型：', type(data))
print('样本数量：', len(data))

# 打印前若干个 key 和对应 value 的详细信息
for i,(k,v) in enumerate(data.items()):
    if i >= 10: break
    print('---- entry', i, '----')
    print('key:', repr(k))
    describe(v, indent=1, max_depth=3)
    print()

# 统计一下所有 value 的类型出现频率
type_count = {}
for v in data.values():
    t = type(v)
    type_count[t] = type_count.get(t,0) + 1
print('value 类型分布:')
pprint.pprint(type_count)

# 如果 value 中包含可以转换为 numpy 数组的内容,
# 还可以把它们的形状、最小/最大/均值打印出来:
for i,(k,v) in enumerate(data.items()):
    if hasattr(v, 'shape') or isinstance(v, (list, tuple, np.ndarray)):
        arr = np.array(v)
        print(f'[{i}]', k, '->', arr.shape, arr.dtype,
              'min', arr.min() if arr.size else None,
              'max', arr.max() if arr.size else None,
              'mean', arr.mean() if arr.size else None)
    if i>=5: 
        print(arr)
        break