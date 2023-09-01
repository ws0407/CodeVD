# import graphviz

# # 读取本地的.dot文件
# dot_file_path = 'output/pdg/1-pdg.dot'
#
# # 转换为Digraph对象
# graph = graphviz.Source.from_file(dot_file_path)
#
# # 可以对graph进行操作，例如渲染为图像或输出为其他格式
# # graph.render('output/1-pdg', format='pdf')
# graph.format = 'dot'
# graph.view()

import graphviz

# 从DOT文件创建图形对象
# 创建一个空白的有向图
graph = graphviz.Digraph()

# 添加节点
graph.node('A')
graph.node('B')
graph.node('C')
graph.node('D')

# 添加边
graph.edge('A', 'B')
graph.edge('B', 'C')
graph.edge('C', 'D')
graph.edge('D', 'A')


# 获取DOT文件内容
dot_content = graph.source

# 解析DOT内容，提取节点信息
lines = dot_content.split("\n")
for line in lines:
    line = line.strip()
    if line.startswith(("node", "subgraph")):
        parts = line.split("[")
        node_name = parts[0].strip()
        attributes = parts[1].strip().strip("];")

        print("Node:", node_name)

        # 解析节点属性
        attribute_pairs = attributes.split(",")
        for attr_pair in attribute_pairs:
            attr_name, attr_value = attr_pair.strip().split("=")
            print(f"Attribute: {attr_name.strip()} = {attr_value.strip()}")

        print("---")