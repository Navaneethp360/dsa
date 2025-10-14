#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
static bool cmp(const vector<int>& a, const vector<int>& b)
{
    return a[1]<b[1];
}
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(),intervals.end(),cmp);
        int count=0; int prevEnd=intervals[0][1];
        for(int i=1;i<intervals.size();++i)
        {
            if(intervals[i][0]<prevEnd)
            count++;
            else
            prevEnd=intervals[i][1];
        }
        return count;
    }
};


Given an integer array nums, return the length of the longest strictly increasing subsequence.

Example 1:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

Example 2:
Input: nums = [0,1,0,3,2,3]
Output: 4

Example 3:
Input: nums = [7,7,7,7,7,7,7]
Output: 1

Constraints:
1 <= nums.length <= 2500
-104 <= nums[i] <= 104
Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?

//brute

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int count=1,maxC=1,n=nums.size(),curr;
        for(int i=0;i<n;++i)
        {
            curr=nums[i];
            count=1;
            for(int j=i;j<n;++j)
            {
                if(nums[j]>curr)
                {
                    count++;
                    curr=nums[j];
                    maxC=max(maxC,count);
                }
            }
        }
        return maxC;
    }
};

//optimized

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n=nums.size();
        vector<int>sub;
        for(int i=0;i<n;++i)
        {
            if(sub.size()==0||sub.back()<nums[i])
            sub.push_back(nums[i]);
            else
            {
                auto idx=lower_bound(sub.begin(),sub.end(),nums[i]);
                *it=nums[i];
            }
        }
        return sub.size();
    }
};

class Solution {
public:
struct Compare {
    bool operator()(ListNode* a, ListNode* b) {
        return a->val > b->val; 
    }
};
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*,vector<ListNode*>,Compare>pq;
        for(auto l:lists)
        {
            if(l!=nullptr) pq.push(l);
        }
        ListNode dummy(0);
        ListNode* tail=&dummy;
        while(!pq.empty())
        {
            ListNode* node=pq.top();
            pq.pop();
            tail->next=node;
            tail=node;
            if(node->next) pq.push(node->next);
        }
        return dummy.next;
    }
};




class Solution {
public:
struct cmp {
    bool operator()(ListNode*a,ListNode*b)
{
    return a->val>b->val;
}
};
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*,vector<ListNode*>,cmp>pq;
        for(auto l:lists)
        {
            if(l)pq.push(l);
        }
        ListNode dummy(0);
        ListNode* tail=&dummy;
        while(!pq.empty())
        {
            ListNode* node=pq.top();
            pq.pop();
            tail->next=node;
            tail=tail->next;
            if(node->next)pq.push(node->next);
        }
        return dummy.next;
    }
};

You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the lists nodes. Only nodes themselves may be changed.

Input: head = [1,2,3,4]
Output: [1,4,2,3]

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]


class Solution {
public:
    void reorderList(ListNode* head) {
        //find middle
        ListNode* slow=head,*fast=head;
        while(fast && fast->next)
        {
            slow=slow->next;
            fast=fast->next->next;
        }
        ListNode* second=slow->next;
        slow->next=nullptr;
        ListNode* prev=nullptr,*next;
        //reverse the second part
        while(second)
        {
            next=second->next;
            second->next=prev;
            prev=second;
            second=next;
        }
        second=prev;
        ListNode*first=head;
        //interlink
        while(second)
        {
            ListNode *t1=first->next;
            ListNode *t2=second->next;
            first->next=second;
            second->next=t1;
            first=t1;
            second=t2;
        }
    }
};



class Solution {
public:
    void reorderList(ListNode* head) {
        ListNode *slow=head,*fast=head;
        while(fast && fast->next)
        {
            slow=slow->next;
            fast=fast->next->next;
        }
        ListNode *second=slow->next;
        slow->next=nullptr;

        ListNode*prev=nullptr;
        while(second)
        {
            ListNode *next=second->next;
            second->next=prev;
            prev=second;
            second=next;
        }
        second=prev;
        ListNode *first=head;

        while(second)
        {
            ListNode* t1=first->next;
            ListNode*t2=second->next;
            first->next=second;
            second->next=t1;
            first=t1;
            second=t2;
        }
    }
};


Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.

You may not alter the values in the lists nodes, only nodes themselves may be changed.

 

Example 1:


Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]
Example 2:


Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
 

Constraints:

The number of nodes in the list is n.
1 <= k <= n <= 5000
0 <= Node.val <= 1000
 

Follow-up: Can you solve the problem in O(1) extra memory space?

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        int n=0;
        ListNode* it=head;
        while(it)
        {
            n++;
            it=it->next;
        }
        int batch=n/k;
        ListNode* newHead = nullptr;
        ListNode* prevTail = nullptr;
        for(int i=0;i<batch;++i)
        {   
            ListNode*prev=nullptr,*next=nullptr,*batchHead=head,*curr=head;
            for(int j=0;j<k;++j)
            {
                next=curr->next;
                curr->next=prev;
                prev=curr;
                curr=next;
            }
            if (i == 0) newHead = prev;
            else prevTail->next = prev;
            batchHead->next = curr;
            prevTail = batchHead; 
            head = curr;
        }
        return newHead;
    }
};


struct Node
{
    int key;
    int value;
    Node* prev;
    Node* next;

    Node(int k, int v) {
        key = k;
        value = v;
        prev = nullptr;
        next = nullptr;
    }
};
class LRUCache {
    private:
    Node *head;
    Node *tail;
    int capacity;
    unordered_map<int,Node*>mp;

     void addNode(Node* node)
     {
        node->prev=head;
        node->next=head->next;
        head->next->prev=node;
        head->next=node;
     }  
     void remNode(Node* node)
     {
        node->prev->next=node->next;
        node->next->prev=node->prev;
     } 
     void moveToHead(Node* node)
     {
        remNode(node);
        addNode(node);
     }
     Node* popTail()
     {
        Node* res=tail->prev;
        remNode(res);
        return res;
     }
public:
    LRUCache(int capacity) {
        this->capacity=capacity;
        head=new Node(0,0);
        tail=new Node(0,0);
        head->next=tail;
        tail->prev=head;
    }
    
    int get(int key) {
        if(mp.find(key)==mp.end()) return -1;
        Node* node=mp[key];
        moveToHead(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if(mp.find(key)!=mp.end())
        {
            Node* node=mp[key];
            node->value=value;
            moveToHead(node);
        }
        else
        {
            Node* node=new Node(key,value);
            addNode(node);
            mp[key]=node;
            if(mp.size()>capacity)
            {
                Node *temp=popTail();
                mp.erase(temp->key);
                delete temp;
            }
        }
    }
};



class Solution {
public:
    Node* cloneGraph(Node* node) {
     unordered_map<Node*,Node*>mp;
     return clone(node,mp);   
    }
    Node* clone(Node* node,unordered_map<Node*,Node*>&mp)
    {
        if(node==nullptr) return nullptr;
        if(mp[node]) return mp[node];
        Node* cl=new Node(node->val);
        mp[node]=cl;
        for(auto neighbor:node->neighbors)
        {
            cl->neighbors.push_back(clone(neighbor,mp));
        }
        return cl;
    }
};



class Solution {
public:
    Node* cloneGraph(Node* node) {
        unordered_map<Node*,Node*>mp;
        return clone(node,mp);
    }
    Node* clone(Node* node,unordered_map<Node*,Node*>&mp)
    {
        if(node==nullptr) return nullptr;
        if(mp[node]) return mp[node];
        Node* cl=new Node(node->val);
        mp[node]=cl;
        for(auto neighbor:node->neighbors)
        {
            cl->neighbors.push_back(clone(neighbor,mp));
        }
        return cl;
    }
};













struct Node
{
    int key;
    int value;
    Node* prev;
    Node* next;

    Node(int k, int v) {
        key = k;
        value = v;
        prev = nullptr;
        next = nullptr;
    }
};
class LRUCache {
    private:
    Node *head;
    Node *tail;
    int capacity;
    unordered_map<int,Node*>mp;

     void addNode(Node* node)
     {

     }  
     void remNode(Node* node)
     {

     } 
     void moveToHead(Node* node)
     {

     }
     Node* popTail()
     {
        
     }
public:
    LRUCache(int capacity) {
        
    }
    
    int get(int key) {
        
    }
    
    void put(int key, int value) {
        
    }
};


class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>>graph(numCourses);
        for(auto p:prerequisites)
        {
            graph[p[1]].push_back(p[0]);
        }
        vector<int>visited(numCourses,0);
        for(int i=0;i<numCourses;++i)
        {
            if(dfs(i,graph,visited)) return false;
        }
        return true;
    }
    bool dfs(int idx,vector<vector<int>>&graph,vector<int>&visited)
    {
        if(visited[idx]==1) return true;
        if(visited[idx]==2) return false;
        visited[idx]=1;
        for(auto c:graph[idx])
        {
            if(dfs(c,graph,visited)) return true;
        }
        visited[idx]=2;
        return false;
    }
};















There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.
Example 2:

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
 

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>>adj(numCourses);
        for(auto [course,preq]:prerequisites)
        {
            adj[preq].push_back(course);
        }    
        vector<int>visited(numCourses,0);
        for(int i=0;i<numCourses;++i)
        {
            dfs(i,adj,visited) return false;
        }
        return true;
    }
    bool dfs(int i,vector<vector<int>>&adj,vector<int>&visited)
    {
        if(visited[i]==1) return true;
        if(visited[i]==2) return false;
        visited[i]=1; 
        for(auto c:adj[i])
        {
            if(dfs(c,adj,visited)) return true;
        }
        visited[i]=2;
        return fasle;
    }
};



class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<pair<int,int>>>graph(n+1);
        for(auto &t:times)
        {
            graph[t[0]].push_back({t[1],t[2]});
        }
        vector<int>dist(n+1,INT_MAX);
        dist[k]=0;
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq;
        pq.push({0,k});
        while(!pq.empty())
        {
            auto [d,node]=pq.top();pq.pop();
            if(d>dist[node]) continue;
            for(auto &ed:graph[node])
            {
                if(dist[ed.first]>d+ed.second)
                {
                    dist[ed.first]=d+ed.second;
                    pq.push({dist[ed.first],ed.first});
                }
            }
        }
        int ans=0;
        for(int i=1;i<=n;++i)
        {
            if(dist[i]==INT_MAX) return -1;
            ans=max(ans,dist[i]);
        }
        return ans;
    }
};



You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.

We will send a signal from a given node k. Return the minimum time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.

 

Example 1:


Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2
Example 2:

Input: times = [[1,2,1]], n = 2, k = 1
Output: 1
Example 3:

Input: times = [[1,2,1]], n = 2, k = 2
Output: -1
 

Constraints:

1 <= k <= n <= 100
1 <= times.length <= 6000
times[i].length == 3
1 <= ui, vi <= n
ui != vi
0 <= wi <= 100
All the pairs (ui, vi) are unique. (i.e., no multiple edges.)


class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<pair<int,int>>>&graph(n+1);
        for(auto &t:times)
        {
            int from=t[0],to=t[1],time=[2];
            graph[from].push_back({to,time});
        }
        vector<int>dist(n+1,INT_MAX);
        dist[k]=0;
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq;
        pq.push({0,k});
        while(!pq.empty())
        {
            auto [d,node]=pq.top();pq.pop();
            if(d>dist[node]) continue;
            for(auto &ed:graph[node])
            {
                if(dist[ed.first]>d+ed.second)
                {
                    dist[ed.first]=d+ed.second;
                    pq.push({dist[ed.first],ed.first});
                }
            }
        }
        int minTime=-1;
        for(int i=1;i<=n;++i)
        {
            if(dist[i]==INT_MAX) return -1;
            minTime=max(minTime,dist[i]);
        }
        return minTime;
    }
};















struct TrieNode {
    TrieNode* children[26];
    string word; // store the complete word at the end node
    TrieNode() {
        for(int i=0;i<26;++i) 
        children[i]=nullptr;
        
        word = "";
    }
};

class Solution {
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        vector<string> res;
        if (board.empty() || board[0].empty() || words.empty()) return res;

        TrieNode* root = buildTrie(words);
        int m = board.size(), n = board[0].size();

        // start DFS from every cell
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                dfs(board, i, j, root, res);
            }
        }

        return res;
    }

private:
    TrieNode* buildTrie(const vector<string>& words) {
        TrieNode* root = new TrieNode();
        for (const string& word : words) {
            TrieNode* node = root;
            for (char c : word) {
                int idx = c - 'a';
                if (!node->children[idx])
                    node->children[idx] = new TrieNode();
                node = node->children[idx];
            }
            node->word = word; // mark end of word
        }
        return root;
    }

    void dfs(vector<vector<char>>& board, int i, int j, TrieNode* node, vector<string>& res) {
        char c = board[i][j];
        int idx = c - 'a';
        if (!node->children[idx]) return;

        node = node->children[idx];
        if (!node->word.empty()) { // found a word
            res.push_back(node->word);
            node->word = ""; // avoid duplicates
        }

        board[i][j] = '#'; // mark visited
        int m = board.size(), n = board[0].size();
        int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};
        for (auto& d : dirs) {
            int ni = i + d[0], nj = j + d[1];
            if (ni >= 0 && ni < m && nj >= 0 && nj < n && board[ni][nj] != '#') {
                dfs(board, ni, nj, node, res);
            }
        }
        board[i][j] = c; // backtrack
    }
};




//Account Merge
class Solution {
public:
        unordered_map<string,string> parent;
        string find(string x)
        {
            if(parent[x]!=x)
            parent[x]=find(parent[x]);
            return parent[x];
        }
        void unionSet(string x,string y)
        {
            string rootX=find(x);
            string rootY=find(y);
            if(rootX!=rootY)
            parent[rootY]=rootX;
        }
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        unordered_map<string,string>etn;
        for(auto &account:accounts)
        {
            string name=account[0]; string first=account[1];
            for(int i=1;i<account.size();++i)
            {
                string email=account[i];
                if(parent.find(email)==parent.end())
                parent[email]=email;
                unionSet(first,email);
                etn[email]=name;
            }
        }
        unordered_map<string,vector<string>> grp;
        for(auto &g:parent)
        {
            string email=g.first;
            string root=find(email);
            grp[root].push_back(email);
        }
        vector<vector<string>>merged;
        for(auto &acc:grp)
        {
            string first=acc.first;
            vector<string> emails=acc.second;
            sort(emails.begin(),emails.end());
            vector<string>a;
            a.push_back(etn[first]);
            a.insert(a.end(),emails.begin(),emails.end());
            merged.push_back(a);
        }
        return merged;
    }
};



Given a list of accounts where each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.

 

Example 1:

Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Explanation:
The first and second Johns are the same person as they have the common email "johnsmith@mail.com".
The third John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.


class Solution {
public:
    unordered_map<string,string>parent;
    string find(string x)
    {
        if(parent[x]!=x)
        parent[x]=find(parent[x]);
        return parent[x];
    }
    void unionSet(string x,string y)
    {
        string rootX=find(x);
        string rootY=find(y);
        if(rootX!=rootY) parent[rootY]=rootX;
    }
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        unordered_map<string,string>etn;
        for(auto account:accounts)
        {
            string name=account[0];
            string first=account[1];
            for(int i=1;i<account.size();++i)
            {
                string email=account[i];
                if(parent.find(email)==parent.end())
                parent[email]=email;
                unionSet(email,first);
                etn[email]=name;
            }
        }
        unordered_map<string,vector<string>>groups;
        for(auto p:parent)
        {
            string email=p.first;
            string root=find(email);
            group[root].push_back(email);
        }
        vector<vector<string>>merged;
        for(auto g:groups)
        {
            string root=g.first;
            vector<string>emails=g.second;
            sort(emails.begin(),emails.end());
            vector<string>account;
            account.push_back(etn[root]);
            account.insert(accounts.end(),emails.begin(),emails.end());
            merged.push_back(account);
        }
        return merged;
    }
};


class Solution {
public:
    vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
        int n=graph.size();
        vector<int> visited(n,0);
        for(int i=0;i<n;++i)
        {
            dfs(i,graph,visited);
        }
        vector<int>ans;
        for(int it=0;it<n;++it)
        {
            if(visited[it]==2)
            ans.push_back(it);
        }
        return ans;
    }
    bool dfs(int idx,vector<vector<int>>&graph,vector<int>&visited)
    {
        if(visited[idx]==1) return true;
        if(visited[idx]==2) return false;
        visited[idx]=1;
        for(auto neighbor:graph[idx])
        {
            if(dfs(neighbor,graph,visited)) return true;
        }
        visited[idx]=2;
        return false;
    }
};



There is a directed graph of n nodes with each node labeled from 0 to n - 1. The graph is represented by a 0-indexed 2D integer array graph where graph[i] is an integer array of nodes adjacent to node i, meaning there is an edge from node i to each node in graph[i].

A node is a terminal node if there are no outgoing edges. A node is a safe node if every possible path starting from that node leads to a terminal node (or another safe node).

Return an array containing all the safe nodes of the graph. The answer should be sorted in ascending order.

 

Example 1:

Illustration of graph
Input: graph = [[1,2],[2,3],[5],[0],[5],[],[]]
Output: [2,4,5,6]
Explanation: The given graph is shown above.
Nodes 5 and 6 are terminal nodes as there are no outgoing edges from either of them.
Every path starting at nodes 2, 4, 5, and 6 all lead to either node 5 or 6.
Example 2:

Input: graph = [[1,2,3,4],[1,2],[3,4],[0,4],[]]
Output: [4]
Explanation:
Only node 4 is a terminal node, and every path starting at node 4 leads to node 4.

class Solution {
public:
    vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
        int n=graph.size();
        vector<vector<int>>adj(n);
        for(int i=0;i<n;++i) //O(n)
        {
            for(auto g:graph[i]) //O(avg graph size)
            {
                adj[g].push_back(i);
            }
        }
        vector<int>deg(n);
        for(int i=0;i<n;++i) //O(n)
        {
            deg[i]=graph[i].size();
        }
        vector<int>ans;
        queue<int>q;
        for(int i=0;i<n;++i) //O(n)
        {
            if(deg[i]==0)
            q.push(i);
        }
         while(!q.empty())
         {
            int el=q.front();q.pop();
            ans.push_back(el);
            for(auto g:adj[el])
            {
                deg[g]--;
                if(deg[g]==0)
                q.push(g);
            }
         }
         sort(ans.begin(),ans.end()); 
         return ans;
    }
};



class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int left,right,sum,target,n=nums.size();
        vector<vector<int>> ans;
        sort(nums.begin(),nums.end());
        for(int i=0;i<n-2;++i)
        {
            if(i>0 && nums[i]==nums[i-1]) continue;
            target=-nums[i];
            left=i+1;
            right=n-1;
            while(left<right)
            {
                sum=nums[left]+nums[right];
                if(sum==target)
                {
                    ans.push_back({nums[i],nums[left],nums[right]});
                    left++; right--;
                    while(left<right&&nums[left]==nums[left-1]) left++;
                    while(left<right&&nums[right]==nums[right+1])right--;
                }
                else if(sum<target) left++;
                else right--;
            }
        }
        return ans;
    }
};


class Solution {
public:
    int maxArea(vector<int>& height) {
        int left=0,n=height.size(),right=n-1,maxar=0,minh=0,ar=0,w=0;
        while(left<right)
        {
            minh=min(height[left],height[right]);
            w=right-left;
            ar=minh*w;
            maxar=max(maxar,ar);
            if(height[left]<height[right])
            left++;
            else
            right--;
        }
        return maxar;
    }
};


class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n=s.size(),left=0,right=0,leng=0;
        unordered_map<char,int>mp;
        for(right=0;right<n;++right)
        {
            if(mp.count(s[right]))
            {
                while(mp.count(s[right]))
                {
                    mp[s[left]]--;
                    if(mp[s[left]]==0) mp.erase(s[left]);
                    left++;
                }
            }
            mp[s[right]]++;
            leng=max(leng,right-left+1);            
        }
        return leng;
    }
};


class Solution {
public:
    string minWindow(string s, string t) {
        vector<int> mp(256,0);
        for(char ch:t)
        mp[ch]++;
        int count=0,left=0,right=0,minlen=INT_MAX,sind=-1,n=t.size(),m=s.size();
        while(right<m)
        {
            mp[s[right]]--;
            if(mp[s[right]]>=0)
            count++;
            while(count==n)
            {
                if(right-left+1<minlen)
                {
                    minlen=right-left+1;
                    sind=left;
                }
                mp[s[left]]++;
                if(mp[s[left]]>0)
                count--;
                left++;
            }
            right++;
        }
        if(minlen==INT_MAX) return "";
        return s.substr(sind,minlen);
    }
};


class Solution {
public:
    bool isSafe(vector<string>&board,int row,int col,int n)
    {
        for(int i=0;i<n;++i)
        {
            if(board[row][i]=='Q') return false;
        }
        for(int i=0;i<n;++i)
        {
            if(board[i][col]=='Q') return false;
        }
        for(int i=row,j=col;i>=0&&j>=0;--i,--j)
        {
            if(board[i][j]=='Q') return false;
        }
        for(int i=row,j=col;i>=0&&j<n;--i,++j)
        {
            if(board[i][j]=='Q') return false;
        }
        return true;
    }
    void nQueens(vector<string>&board,vector<vector<string>>&ans,int n,int row)
    {
        if(row==n)
        {
            ans.push_back(board);
            return;
        }
        for(int i=0;i<n;++i)
        {
            if(isSafe(board,row,i,n))
            {
                board[row][i]='Q';
                nQueens(board,ans,n,row+1);
                board[row][i]='.';
            }
        }
    }
    vector<vector<string>> solveNQueens(int n) {
        vector<string>board(n,string(n,'.'));
        vector<vector<string>>ans;
        nQueens(board,ans,n,0);
        return ans;
    }
};


class Solution {
public:
    void solveSudoku(vector<vector<char>>& board) {
        solve(board);
    }
    bool solve(vector<vector<char>>& board)
    {
        for(int i=0;i<9;++i)
        {
            for(int j=0;j<9;++j)
            {
                if(board[i][j]=='.')
                {
                    for(char digit='1';digit<='9';++digit)
                    {
                        if(isValid(board,digit,i,j))
                        {
                            board[i][j]=digit;
                            if(solve(board)) return true;
                            board[i][j]='.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
    bool isValid(vector<vector<char>>& board,char &ch,int &row,int &col)
    {
        for(int i=0;i<9;++i)
        {
            if(board[row][i]==ch) return false;
            if(board[i][col]==ch) return false; 
        }
        int boxrow=(row/3)*3;
        int boxcol=(col/3)*3;
        for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
        if(board[boxrow+i][boxcol+j]==ch) return false;
        return true;
    }
};



//merge k sorted lists

class Solution {
public:
struct cmp {
    bool operator()(ListNode*a,ListNode*b)
{
    return a->val>b->val;
}
};
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*,vector<ListNode*>,cmp>pq;
        for(auto l:lists)
        {
            if(l)pq.push(l);
        }
        ListNode dummy(0);
        ListNode* tail=&dummy;
        while(!pq.empty())
        {
            ListNode* node=pq.top();
            pq.pop();
            tail->next=node;
            tail=tail->next;
            if(node->next)pq.push(node->next);
        }
        return dummy.next;
    }
};


struct cmp{
    bool operator()(ListNode*a,ListNode*b)
    {
        return a->val>b->val;
    }
};
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*,vector<ListNode*>,cmp>pq;
        for(auto l:lists)
        {
            if(l).pq.push(l);
        }
        ListNode dummy(0);
        ListNode* tail=&dummy;
        while(!pq.empty())
        {
            ListNode* node=pq.top();
            pq.pop();
            tail->next=node;
            tail=tail->next;
            if(node->next) pq.push(node->next);
        }
        return dummy.next;
    }
};





A. Arrays / Strings

Two Pointers → e.g. 3Sum, Container With Most Water

Sliding Window → Longest Substring Without Repeating, Minimum Window Substring

Prefix Sum / Difference Array → Subarray Sum Equals K

Kadane’s Algorithm → Maximum Subarray Sum

💡 Focus: subarray/subsequence patterns, prefix usage, variable window movement logic.

B. Hashing / Maps / Sets

Frequency counting, hash-based lookups → Two Sum, Group Anagrams

Detecting duplicates / subarrays with target sums

When to use unordered_map vs ordered_map

C. Linked Lists

Fast/slow pointers → detect cycle, find middle, intersection

Reversal (full or partial)

Merging sorted lists

Copy list with random pointer

💡 Focus on pointer handling, not syntax.

D. Stacks & Queues

Monotonic Stack / Next Greater Element

Valid Parentheses

Daily Temperatures, Stock Span

Min Stack / Stack with supporting data

💡 Know when to use stack for sequence scanning (increasing/decreasing trend).

E. Trees & Binary Search Trees

DFS traversals (pre/in/postorder)

BFS / Level order traversal

Height / Diameter / Balance check

Lowest Common Ancestor

BST insert, validate, kth smallest

💡 Practice recursion intuition: “What do I get from left and right children?”

F. Graphs

DFS/BFS traversal

Topological Sort (Kahn + DFS) ✅ you just mastered this

Cycle detection (directed + undirected)

Connected components (DFS / Union-Find)

Shortest Path → Dijkstra, BFS on unweighted

💡 Be crystal-clear when to use BFS vs DFS, and how visited[] behaves.

G. Dynamic Programming

1D DP: Fibonacci, Climbing Stairs, House Robber, Jump Game

2D DP: Grid paths, Min path sum, Unique paths with obstacles

Knapsack patterns: subset sum, partition equal subset

String DP: LCS, Edit Distance, Palindromic Subsequence

💡 Framework:
choice → recurrence → memoization → tabulation → space optimization

H. Greedy

Interval Scheduling → Activity Selection, Merge Intervals

Minimum Platforms / Meeting Rooms

Huffman, Kruskal, Prim (if they go advanced)

💡 Prove greediness by local optimal → global optimal.

I. Binary Search

Normal binary search (edge conditions)

Search insert position

Rotated sorted array

Search in matrix

Binary search on answer → Minimum days, capacity to ship

J. Heaps / Priority Queues

Kth largest / smallest

Top K frequent elements

Merge K sorted lists

Median in stream

K. Backtracking

Subsets / Permutations / Combination Sum

N-Queens

Sudoku Solver

💡 Focus on decision tree + pruning intuition.

L. Graph + Topo Sort Extensions

Eventual Safe States ✅

Course Schedule

Alien Dictionary

Detect cycles via indegree logic
