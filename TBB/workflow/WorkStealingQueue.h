#include <deque>
#include <vector>
#include <memory>
#include <iostream>
#include <tbb/flow_graph.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

using namespace tbb::flow;

// 树节点数据结构
struct TreeNode {
    int value;
    std::vector<TreeNode*> children;
    
    TreeNode(int val) : value(val) {}
};

// 工作流任务基类
class WorkflowTask {
public:
    virtual void execute() = 0;
    virtual ~WorkflowTask() = default;
};

// 示例任务类型
class ProcessingTask : public WorkflowTask {
    int data;
public:
    ProcessingTask(int d) : data(d) {}
    
    void execute() override {
        std::cout << "Processing data: " << data 
                  << " on thread: " << std::this_thread::get_id() << "\n";
    }
};

// 树处理工作流构建器
class TreeWorkflow {
    using NodePtr = std::shared_ptr<WorkflowTask>;
    
    graph& flow_graph;
    std::vector<continue_node<continue_msg>>& merge_points;
    
public:
    TreeWorkflow(graph& g, std::vector<continue_node<continue_msg>>& mp)
        : flow_graph(g), merge_points(mp) {}

    void build(TreeNode* root) {
        // 创建当前节点任务
        auto task_node = make_node(root->value);
        
        // 递归处理子节点
        std::vector<continue_node<continue_msg>> child_nodes;
        for(auto child : root->children) {
            TreeWorkflow child_flow(flow_graph, merge_points);
            child_flow.build(child);
            child_nodes.push_back(child_flow.get_start_node());
        }

        // 建立依赖关系
        if(!child_nodes.empty()) {
            make_edge(task_node, create_join_node(child_nodes));
        }
    }

private:
    continue_node<continue_msg> make_node(int value) {
        return continue_node<continue_msg>(flow_graph, 
            [=](continue_msg) {
                ProcessingTask task(value);
                task.execute();
            });
    }

    continue_node<continue_msg> create_join_node(
        const std::vector<continue_node<continue_msg>>& children) 
    {
        // 创建合并节点
        auto join_node = continue_node<continue_msg>(flow_graph,
            [](continue_msg) {});

        // 建立合并关系
        for(auto& child : children) {
            make_edge(child, join_node);
        }
        
        // 添加到全局合并点
        merge_points.push_back(join_node);
        return join_node;
    }
};

// DAG工作流构建器
class DagWorkflow {
    graph flow_graph;
    std::vector<continue_node<continue_msg>> merge_points;

public:
    void build_and_run(TreeNode* root) {
        // 构建树状工作流
        TreeWorkflow tree_flow(flow_graph, merge_points);
        tree_flow.build(root);

        // 添加全局最终合并节点
        auto final_node = continue_node<continue_msg>(flow_graph,
            [](continue_msg) {
                std::cout << "All tasks completed\n";
            });

        for(auto& mp : merge_points) {
            make_edge(mp, final_node);
        }

        // 启动工作流
        broadcast_node<continue_msg> start_node(flow_graph);
        make_edge(start_node, tree_flow.get_start_node());
        start_node.try_put(continue_msg());
        flow_graph.wait_for_all();
    }
};

// 示例树结构构建
TreeNode* build_sample_tree() {
    TreeNode* root = new TreeNode(0);
    
    for(int i=1; i<=3; ++i) {
        TreeNode* child = new TreeNode(i);
        root->children.push_back(child);
        
        for(int j=1; j<=2; ++j) {
            TreeNode* grandchild = new TreeNode(i*10+j);
            child->children.push_back(grandchild);
        }
    }
    return root;
}