CPU
 └── launch kernel
      └── Grid（二维）
           └── Block（16×16）
                └── Thread（1 个 C[row][col]）



Block
 ├── 所有 thread 加载 As / Bs
 ├── __syncthreads()
 ├── 所有 thread 使用 As / Bs
 ├── __syncthreads()
 ├── 下一轮 t


Grid
 ├── Block(0,0) → C[0..15][0..15]
 ├── Block(1,0) → C[0..15][16..31]
 ├── Block(0,1) → C[16..31][0..15]
 ├── ...



block 之间：
无通信
无同步
顺序不确定

一个 thread = 算一个 C 元素
一个 block = 协作算一个 tile
Grid = 覆盖整个矩阵
K 维度靠时间循环完成
