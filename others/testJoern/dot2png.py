import graphviz

# 读取本地的.dot文件
dot_file_path = '/data/data/ws/CodeVD/others/testJoern/output/pdg/1-pdg.dot'
with open(dot_file_path, 'r') as file:
    dot_data = file.read()

# 转换为Digraph对象
graph = graphviz.Source(dot_data)

# 可以对graph进行操作，例如渲染为图像或输出为其他格式
graph.render('/data/data/ws/CodeVD/others/testJoern/output/pdg/1-pdg.png', format='png')