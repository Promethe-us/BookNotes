## git使用

```git
'查看所有帮助'
git help -a

‘设置用户信息'
‘如需修改，在git后面加上unset即可’
git config --global user.name 'wuke’
git config --global user.email 'wuke20010216@outlook.com'

'查看用户信息'
cat ~/.gitconfig

'查看git文件状态'
git status

‘建立好文件夹以后要初始化’
git init

'添加全部文件'
git add .

'提交修改'
git commit

'查看提交记录'
git log
git log --oneline

'查看修改的内容-上一版本与现在版本的区别'
git diff test.py
'查看repository与暂存区的区别'
git diff --staged

'重命名或者移动文件'
git mv test.py testv.py

'删除文件'
git rm py/track.py
'删除文件夹'
git rm -r py

'在commit之前恢复删除'
git checkout HEAD -- test.py

'恢复至历史版本'
git log --oneline 得到id号
git revert id

‘提交之后HEAD默认指向最新一次，可以通过git reset修改HEAD指向’
git reset --soft id

'查看所有分支'
git branch

‘添加分支’
git branch mobile

'切换分支'
git checkout mobile

'查看所有的分支上的改变'
git log --all

‘查看两个branch的区别’
git diff master..mobile

'分支合并'
(mobile)git merge master

'重命名分支'
git branch -m mobile bugfix
'删除分支'
git branch -d bugfix

'显示最近五条'
git log --oneline -5
'只显示某个作者提交的'
git log --onelione --author='wuke'
'显示对某个文件的修改'
git log --oneline --grep='test.py'
'指定时间范围'
git log --oneline --before='2022-01-01'
git log --oneline --before='oneweek' 

'我们不想让系统跟踪(忽略)一些文件'
git config --global core.excludesfile ~/.gitgore_global

'为每个项目单独创建忽略文件列表'
vim .gitignore

'推送到远程服务器，为项目在远程服务器创建一个版本库，方便协作开发'
‘使用github提供的远程版本库服务’
'执行完了之后origin就指代后面的版本库了'
git reomote add origin https://github.com/promethe-us/bkt.git

git push -u origin

'协作开发'

'下载到自己命名的文件夹'
git clone URL dirname

'提取更新'
git fetch

'将别人的repo复制到自己的repo'
fork

'添加项目贡献者'
serttings -> collaborators -> add collaborator


```



