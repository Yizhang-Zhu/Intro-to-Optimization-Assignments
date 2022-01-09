#include<bits/stdc++.h>
using namespace std;

// 3个中的最大值
int max3_1(int x, int y, int z){
    int temp = max(x, y);
    int maxn = max(temp, z);
    return maxn;
}

// main
int main(){
    // 输入输出文件
    ifstream infile;
    ofstream outfile;
    infile.open("in.txt", ios::in);
    outfile.open("out.txt", ios::trunc);
    if(!infile){
        cout<<"There is nothing in the in.txt file."<<endl;
    }
    // 输入：m行n列
    int m, n;
    infile>>m;
    infile>>n;
    int value[m+1][n+1] = {0};
    int dp[m+1][n+1] = {0}; 
    int path[m+1] = {0};
    int dppath[m+1][n+1] = {1};
    
    for(int i = 1; i<=m; i++){
        for(int j = 1; j<=n; j++){
            infile>>value[i][j];
        }
    }
    // 初始化dp
    for(int i = 1; i<=n; i++){
        dp[1][i] = value[1][i];
    }
    // DP，注意边界
    dp[1][1] = value[1][1];
    for(int i = 2; i<=m; i++){
        for(int j = 1; j<=n; j++){
            if(j == 1){
                // 行首，j-1不存在
                dp[i][j] = max(dp[i-1][j], dp[i-1][j+1]) + value[i][j];
                if(dp[i][j] == dp[i-1][j]+value[i][j]){
                    dppath[i][j] = j;
                }else{
                    dppath[i][j] = j+1;
                }
            }else if(j == n){
                // 行尾，j+1不存在
                dp[i][j] = max(dp[i-1][j-1], dp[i-1][j]) + value[i][j];
                if(dp[i][j] == dp[i-1][j-1]+value[i][j]){
                    dppath[i][j] = j-1;
                }else{
                    dppath[i][j] = j;
                }
            }else{
                dp[i][j] = max3_1(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]) + value[i][j];
                if(dp[i][j] == dp[i-1][j-1]+value[i][j]){
                    dppath[i][j] = j-1;
                }else if(dp[i][j] == dp[i-1][j]+value[i][j]){
                    dppath[i][j] = j;
                }else{
                    dppath[i][j] = j+1;
                }
            }
        }
    }

    // for(int i = 1; i<m; i++){
    //     int m = i+1;
    //     if(dp[m][n-1] == max3_1(dp[m][n-1], dp[m][n], dp[m][n+1])){
    //         path[m] = n-1;
    //         n--;
    //         continue;
    //     }else if(dp[m][n] == max3_1(dp[m][n-1], dp[m][n], dp[m][n+1])){
    //         path[m] = n;
    //         continue;
    //     }else if(dp[m][n+1] == max3_1(dp[m][n-1], dp[m][n], dp[m][n+1])){
    //         path[m] = n+1;
    //         n++;
    //         continue;
    //     }
    // }
    // 结果
    int maxValue = 0;
    int maxValue_index = 1;
    for(int j = 1; j<=n; j++){
        // maxValue = max(maxValue, dp[m][i]);
        if(maxValue < dp[m][j]){
            maxValue = dp[m][j];
            maxValue_index = j;
        }
    }
    cout<<maxValue<<endl;
    path[1] = 1;
    path[m] = maxValue_index;
    for(int i = m; i>1; i--){
        path[i-1] = dppath[i][maxValue_index]; 
        maxValue_index = path[i-1];
    }
    for(int i = 1; i<=m; i++){
        cout<<path[i]<<endl;
    }
}