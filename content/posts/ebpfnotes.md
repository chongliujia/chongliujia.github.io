+++
title = 'eBPF-notes'
date = 2024-01-15T22:50:51-06:00
draft = false
+++

# eBPF

eBPF可以在不修改Linux内核模块的情况下，与内核进行交互。比如进行内核级别的系统监控。

可以利用eBPF建立hook捕捉系统内部情况，比如I/O请求，文件操作，CPU与内存使用情况。

## BPF Maps
一个Map是一个数据结构，它用来访问eBPF程序和用户空间。
Maps能够被用来共享多个eBPF程序或者与用户空间程序和在内核中的eBPF程序通信的数据。

有些典型的使用是：

1. 用户空间写入参数信息通过一个eBPF程序被检索
2. 一个eBPF程序正在存储状态，之后通过另一个eBPF程序被恢复初始状态。
3. 一个eBPF程序写入结果或者指标到一个map中，之后通过一个用户空间程序恢复到初始状态，这个用户空间程序将表示这些结果。

BPF maps是key-value存储模式。
有些map类型被定义为数组，这些数组常常使用4-byte索引作为键值类型。另一些map则是被定义为hash表，这些hash表用一些随机数据类型作为键值类型。

map类型被用来优化某些特别的操作类型，例如first-in-first-out queues, least-recently-used data storage, longest-prefix matching, bloom filters。

有些eBPF map类型拥有特别的对象类型信息。例如，sockmaps和devmaps有sockets和网络设备的信息，它们是通过相关网络eBPF程序重定向流量获得的信息。或者一个程序数组map存储一系列被索引的eBPF程序，它们被用来实现tail calls，即一个程序调用其他程序。

有些map类型有每个CPU的变体，它们意味着内核对每个CPU核信息使用不同的存储块的map。

## Tail Calls
Tail Calls能够调用和执行其他eBPF程序以及替换执行的内容，类似于execve()进行系统调用操作。

可以用bpf_tail_call()的helper函数做Tail Calls：

- 调用bpf_tail_call()时，它将会立即停止当前eBPF程序的执行，并开始执行目标程序。
- 为了减少资源消耗并确保安全性，每个程序在单次调用中的tail call数量被限制在32次。

## eBPF虚拟机
它用来处理eBPF字节码指令(eBPF bytecode instructions)，然后将它们转换成运行在CPU的机器指令集。

eBPF字节码有多个指令组成，这些指令运行在eBPF的虚拟寄存器中。这些eBPF指令集和寄存器模式被设计来映射CPU的架构，以此来直接编译或者中断从字节码到机器码这个过程。

## eBPF寄存器
eBPF虚拟机使用了10个通用寄存器，这些寄存器标成从0到9的序号。另外，10号寄存器被用来做栈帧指针，它只能读不能写。一旦一个eBPF程序开始执行，值被存在这些寄存器中保持状态的顺序。

在执行eBPF程序前，数据(context argument)被加载到1号寄存器中。从函数返回的值被存储在0号存储器中。

在从eBPF代码中调用一个函数之前，这些参数中的函数会存储在1号寄存器到5号寄存器中。

## eBPF的指令
在Linux源码中eBPF的一条指令表示为：  

```
struct bpf_insn {
		__u8 code;        /* opcode */
		__u8 dst_reg:4;   /* dest register */
		__u8 src_reg:4;   /* source register */
		__s16 off;        /* signed offset */
		__s32 imm;        /* signed immediate constant */
};
```

代码说明：

1. 每条指令拥有一个opcode，它用来定义操作指令时，操作的行为是什么，比如写一个值，在一个程序中跳转到一个不同的指令。
2. dst_reg说明了不同的操作可能要用到两个寄存器。
3. off说明，有些操作可能需要偏移值或立即整型值。

bpf_insn 结构的长度为64 bits。有些指令可能需要超过64bits。

当加载到内核时，一个eBPF程序的字节码由一系列bpf_insn结构组成。