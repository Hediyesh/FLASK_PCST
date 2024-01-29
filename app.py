import base64
from flask import Flask, render_template, request
import networkx as nx
import matplotlib.pyplot as plt
import io

app = Flask(__name__)


# in the function below we choose the best neighbor for vertex based on objective = cost +penalty
def select_neighbor(selected_nodes, graph, penalties, selected_edges):
    obj = float('inf')
    vertex = ''
    selected_neighbor = ''
    for node in selected_nodes:
        cost = float('inf')
        penalty = 0
        v = ''
        for neighbor in graph[node]:
            if penalty < penalties[neighbor] and cost > graph[node][neighbor] and (node, neighbor) not in selected_edges and (neighbor, node) not in selected_edges:
                penalty = penalties[neighbor]
                cost = graph[node][neighbor]
                v = neighbor
        if obj > cost + penalty:
            obj = cost + penalty
            vertex = node
            selected_neighbor = v
    return vertex, selected_neighbor


# in the function below we choose the edges that connect the selected nodes chosen by user
def connect(graph, penalties, selected_nodes):
    selected_edges = set()
    new_graph = graph.copy()
    new_penalty = penalties.copy()
    new_selected_nodes = selected_nodes.copy()
    while not is_path(new_graph, selected_nodes):
        v, neighbor = select_neighbor(new_selected_nodes, new_graph, new_penalty, selected_edges)
        selected_edges.add((v, neighbor))
        selected = ''.join(v + neighbor)
        new_graph[selected] = {}
        for node, weight in new_graph[v].items():
            if node != neighbor:
                new_graph[selected][node] = weight
        penalty_sum = new_penalty[v] + new_penalty[neighbor]
        new_penalty[selected] = penalty_sum
        del new_penalty[v]
        del new_penalty[neighbor]
        for i in new_graph:
            if i != selected and i != v and i != neighbor:
                new_graph[i].pop(v)
                new_graph[i].pop(neighbor)
                new_graph[i][selected] = new_graph[selected][i]
        del new_graph[v]
        del new_graph[neighbor]
        new_selected_nodes.append(selected)
        if v in selected_nodes:
            new_selected_nodes.remove(v)
        if neighbor in selected_nodes:
            new_selected_nodes.remove(neighbor)
        for i in new_selected_nodes:
            if i not in new_graph:
                new_selected_nodes.remove(i)
    transformed_edges = []
    edge_set = selected_edges.copy()
    sorted_edges = sorted(edge_set, key=lambda item: len(item[0]))
    for each in sorted_edges:
        transformed_edges.append((each[0][-1], each[1]))
    transformed_edges = [(edge[0], edge[1][0]) if len(edge[1]) > 1 else edge for edge in transformed_edges]
    return transformed_edges


# this is for checking if we have found the path that connects all selected nodes or not
def is_path(graph, selected_nodes):
    nodes = graph.keys()
    length = len(selected_nodes)
    path = False
    for i in nodes:
        k = 0
        for j in selected_nodes:
            if j in i:
                k += 1
        if k == length:
            path = True
    return path


# the first picture showing the graph with selected nodes
def visualize_graph(graph, selected_nodes=None):
    G = nx.Graph(graph)
    pos = nx.circular_layout(G)
    # for selected nodes
    plt.figure(figsize=(8, 6))
    plt.title('Graph with selected nodes (1)')
    # Draw the graph with nodes and edges in blue
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='blue')
    # Highlight selected nodes in red and add a circle around them
    nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_size=500, node_color='red', linewidths=2,
                           edgecolors='black')
    circle_nodes = nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_size=800, node_color='none',
                                          linewidths=2, edgecolors='red')
    circle_nodes.set_edgecolor('red')
    plt.axis('off')
    # Make an image of the graph and convert the plot to an image for embedding in the HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    return f'<img class="center" style="width:100%;height:100%;display:block" src="data:image/png;base64,{encoded_img}" alt="Graph Visualization">'


# the pictures of graph for each step
def draw_graph(graph, selected_nodes, selected_edges, red_edges, row):
    G = nx.Graph(graph)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 6))
    plt.title(f'Next step ({row})')
    # Draw the graph with nodes and edges in blue
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='blue')
    # Highlight the selected edges in red
    nx.draw_networkx_edges(G, pos, edgelist=[selected_edges], edge_color='red', width=2.0)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', width=2.0)
    # Add a circle selected nodes
    nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_size=500, node_color='red', linewidths=2,
                           edgecolors='black')
    circle_nodes = nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_size=800, node_color='none',
                                          linewidths=2, edgecolors='red')
    circle_nodes.set_edgecolor('red')
    plt.axis('off')
    # Make an image of the graph and convert the plot to an image for embedding in the HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    imgres = f'<img class="center" style="width:100%;height:100%;display:block" src="data:image/png;base64,{encoded_img}" alt="Graph Visualization">'
    return imgres


# index
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Extract user input from the form
        selected_nodes = request.form.getlist('selected_nodes')
        first = selected_nodes[0].replace("'", "")
        second = first.split(',')
        selected_nodes = second
        # make the graph
        graph = {
            'A': {'B': int(request.form['AB']), 'D': int(request.form['AD']), 'C': int(request.form['AC'])},
            'B': {'A': int(request.form['AB']), 'C': int(request.form['BC']), 'D': int(request.form['BD'])},
            'C': {'A': int(request.form['AC']), 'B': int(request.form['BC']), 'D': int(request.form['CD'])},
            'D': {'A': int(request.form['AD']), 'B': int(request.form['BD']), 'C': int(request.form['CD'])}
        }
        penalties = {
            'A': int(request.form['penalty_A']),
            'B': int(request.form['penalty_B']),
            'C': int(request.form['penalty_C']),
            'D': int(request.form['penalty_D'])
        }
        penalties_whole = penalties['A'] + penalties['B'] + penalties['C'] + penalties['D']
        # copy of graph
        graph1 = {
            'A': {'B': int(request.form['AB']), 'D': int(request.form['AD']), 'C': int(request.form['AC'])},
            'B': {'A': int(request.form['AB']), 'C': int(request.form['BC']), 'D': int(request.form['BD'])},
            'C': {'A': int(request.form['AC']), 'B': int(request.form['BC']), 'D': int(request.form['CD'])},
            'D': {'A': int(request.form['AD']), 'B': int(request.form['BD']), 'C': int(request.form['CD'])}
        }
        if len(selected_nodes) < 2:
            # if less than 2 nodes are selected, it returns the string below
            result = "Please select nodes!"
            return render_template('result.html', result=result)
        else:
            # make first visualization and add to list
            li_img = []
            visual = visualize_graph(graph, selected_nodes)
            li_img.append(visual)
            paths = connect(graph, penalties, selected_nodes)
            # calculate penalties and costs of each step
            objective = []
            costs = []
            penalty = []
            nodes = []
            costs.append(0)
            penalty.append(penalties_whole)
            objective.append(penalties_whole)
            row = 2
            for i in range(0, len(paths)):
                source, target = paths[i]
                cost = graph[source][target]
                nodes.append(source)
                nodes.append(target)
                p = 0
                red_edges = []
                if i != 0:
                    for j in range(0, i):
                        red_edges.append(paths[j])
                        source2, target2 = paths[j]
                        cost += graph1[source2][target2]
                        nodes.append(source2)
                        nodes.append(target2)
                distinct_list = [item for index, item in enumerate(nodes) if item not in nodes[:index]]
                for each in distinct_list:
                    p += penalties[each]
                costs.append(cost)
                penalty.append((penalties_whole - p))
                obj = cost + (penalties_whole - p)
                objective.append(obj)
                # adding each step's visualization to the list
                li_img.append(draw_graph(graph1, selected_nodes, paths[i], red_edges, row))
                row += 1
            # make a table of costs and penalties
            table_html = "<table class='table table-primary' style='border:1px solid black'><tr><th>Row</th><th>Objective</th><th>Costs</th><th>Penalty</th></tr>"
            r = 0
            for obj, cost, pen in zip(objective, costs, penalty):
                r += 1
                table_html += f"<tr style='background-color:white;color:black;border-spacing: 5px;'><td>{r}</td><td>{obj}</td><td>{cost}</td><td>{pen}</td></tr>"
            table_html += "</table>"
            img_html = ''.join(li_img)
            result = {'li_img': img_html, 'Objective': objective, 'costs': costs, 'penalty': penalty,
                      'table_html': table_html}
            return render_template('result.html', result=result)
    # render the input form page with the result
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
