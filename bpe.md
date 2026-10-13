1.  raw_text = 读取文件内容

2.  初始词汇表 vocab = {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}  // 先塞满256个单字节

3.  特殊标记占位：把 <|endoftext|> 先放进 vocab，给它分配一个 ID（比如 256）

4.  merges = []  // 用来记录合并顺序

    这一步的目标：得到一个字典，key是“单词块”，value是“这个单词块出现了几次”

5.  word_counts = {}  // 格式举例：{ (b'l', b'o', b'w'): 5, (b't', b'h', b'e'): 10 }

6.  把 raw_text 按照特殊标记（<|endoftext|>）切开，得到若干个文档片段。

7.  For 每一个文档片段:
       // 使用 GPT-2 正则表达式（PAT）找出所有匹配的单词
       words = regex.findall(PAT, 文档片段)

       For 每一个 word in words:
           byte_tuple = word.encode('utf-8') 转成字节元组，比如 b'low' -> (108, 111, 119)
           // 注意：这里我们只统计“原始”字节对，暂时还没合并
           word_counts[byte_tuple] += 1  // 计数器加1
       End For
    End For

// 目标：统计所有字节对（A, B）挨在一起出现的总次数
8.  pair_counts = {}  // 格式举例：{ (b'l', b'o'): 7, (b'o', b'w'): 7, (b'e', b's'): 9 }

9.  For (word_tuple, count) in word_counts:
       // 遍历单词里的每两个相邻字节
       For i in 0 到 len(word_tuple) - 2:
           pair = (word_tuple[i], word_tuple[i+1])  // 比如 (108, 111)
           pair_counts[pair] += count  // 加上这个单词出现的总次数
       End For
    End For

    // 当目前的词汇表大小还小于目标大小时，就一直合并

10.  While len(vocab) < V:

     // 10.1 找出频率最高的相邻对
     best_pair = pair_counts 中出现次数最高的那个键（比如 (b't', b'h')）
     // 如果最高次数有并列，选字典序更大的那个（Python默认max就是字典序，记得实现）

     // 10.2 如果没有可合并的了（pair_counts为空），跳出循环
     If best_pair 不存在: Break

     // 10.3 记录这次合并
     merges.append(best_pair)  
     new_token = best_pair[0] + best_pair[1]  // 例如 b't' + b'h' = b'th'
     把 new_token 分配一个新ID，加入 vocab

     // ---------- 最难的部分：只更新受影响的频率（缓存刷新） ----------
     // 注意：千万不能重新扫描整个文本，只改动涉及 best_pair 的地方！

     // 10.4 遍历所有“包含 best_pair 这个组合”的单词
     For (word_tuple, count) in word_counts:
         If word_tuple 里面包含 best_pair (即相邻的这两个字节):
             

         // 假设 word_tuple 为 (..., L, A, B, R, ...)
             // 我们要把 (A, B) 替换成 new_token (AB)
             
             // 步骤 A：删除旧相邻对的计数（因为这些组合现在已经不存在了）
             // 如果 L 存在: pair_counts[(L, A)] -= count
             // 如果 R 存在: pair_counts[(B, R)] -= count
             
             // 步骤 B：增加新相邻对的计数（因为 AB 作为一个整体诞生了）
             // 如果 L 存在: pair_counts[(L, AB)] += count
             // 如果 R 存在: pair_counts[(AB, R)] += count
             
             // 步骤 C：去掉 pair_counts 中计数为 0 或负数的条目（清理垃圾）
             
             // 步骤 D：更新 word_counts 中的这个单词，把 (A, B) 替换为 (AB)
             // 比如 (b'l', b'o', b'w') 遇到合并 (b'l', b'o')，变成 (b'lo', b'w')
         End If
     End For

     End While

11.  返回 merges 和 vocab