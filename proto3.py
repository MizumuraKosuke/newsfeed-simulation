# coding: UTF-8
import math, random, operator, collections
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

#ーーーーーーーーーーーーーーークラスーーーーーーーーーーーーーーー
class User:
    def __init__(self,x,y):
        self.history = []
        self.already_see = 0 #前回までに見たニュースフィードの位置
        self.x = x
        self.y = y
        self.click = []
        self.typ = 0

        #ising weight test
        self.weight = 1



class Content:
    def __init__(self,typ,x,y):
        self.typ = typ #投稿の種類
        self.ising = np.full((x,y), -1) #イジングモデル
        self.cond = [] #時間ごとの記事の情勢を保持



class NewsFeed:
    def __init__(self,x,y,series):
        self.x = x #ユーザーエージェントフィールドのx軸
        self.y = y #ユーザーエージェントフィールドのy軸
        self.users = np.zeros((self.x,self.y), dtype=object) #ユーザーフィールドの枠組み
        self.series = series
        self.contents = [] #コンテンツのタイプごとにオブジェクトを作成し、格納
        self.post_cont = [] #投稿された記事全てを格納。 id, 時間, タイプ, ユーザーx座標, y座標, リツイート数, いいね数, 閲覧数
        self.j = 0.047 #人間関係の強度
        self.stds = [] #ニュースフィードの記事のタイプの標準偏差
        self.hists = [] #ニュースフィードの記事のタイプの度数分布
    
    #アルゴリズム選択関数
    def select_SERIES(self,series,visitor,step):
        if series == "TIME_SERIES":
            return self.TIME_SERIES()
        elif series == "ENGAGE_SERIES":
            return self.ENGAGE_SERIES(visitor, step)
        elif series == "HISTORY_SERIES":
            return self.HISTORY_SERIES(visitor, step)
        else:
            print("そんなアルゴリズムはありません。")
    
    #時系列アルゴリズム
    def TIME_SERIES(self):
        timelist = self.post_cont[:]
        timelist.sort(key=operator.itemgetter(1),reverse=True)
        return timelist
    
    #エンゲージメント率に基づくアルゴリズム
    def ENGAGE_SERIES(self,visitor,step):
        sort = self.TIME_SERIES()
        sort_len = len(sort)
        scoreli = np.linspace(0.02,0,len(sort))
        score_list = [[sort[i][0], scoreli[i]+self.enganement(sort[i],1,1)] for i in range(sort_len)]
        score_list.sort(key=operator.itemgetter(1),reverse=True)
        engage_list = [self.post_cont[self.post_cont[0][0]-i[0]] for i in score_list]
        return engage_list
    
    #閲覧履歴に基づくアルゴリズム
    def HISTORY_SERIES(self,visitor,step):
        sort = self.TIME_SERIES()
        sort_len = len(sort)
        scoreli = np.linspace(0.02,0,len(sort))
        hisnow = np.array(visitor.click)
        for user in hisnow:
            for i in range(sort_len):
                if user[3] == sort[i][3] and user[4] == sort[i][4]:
                    scoreli[i] += 0.003
                    #print('sort[i][2]{}:{}'.format(sort[i][2],scoreli[i]))
        score_list = [[sort[i][0], scoreli[i]] for i in range(sort_len)]
        score_list.sort(key=operator.itemgetter(1),reverse=True)
        history_list = [self.post_cont[self.post_cont[0][0]-i[0]] for i in score_list]
        #print(history_list[:40])
        return history_list

    #記事閲覧
    def look(self, sort, visitor, step, n_step, production):
        see_cont_num = self.see(visitor)
        see_cont = sort[:see_cont_num]
        if production:
            self.add_std(see_cont)
            self.add_hist(see_cont)
        self.add_history(see_cont,visitor)
        rrlook = np.linspace(1,0.5,len(see_cont)) #閲覧確率を上から順に下げていくリスト
        rrtyp = np.array([(1-0.001*abs(visitor.typ-i[2])) for i in see_cont])
        read_rate_list = rrlook * rrtyp
        count = 0
        for cont in see_cont:
            ix = self.post_cont[0][0] - cont[0]
            postlen = len(self.post_cont)
            if postlen > ix:
                self.post_cont[ix][7] += 1 #閲覧数加算
            if random.random() < read_rate_list[count]:
                self.add_click(visitor, cont)
                self.trans(cont,visitor,step)
            count += 1
    
    #状態遷移
    def trans(self, content, visitor, step):
        content_type = content[2]
        vx = visitor.x
        vy = visitor.y
        for i in self.contents:
            if i.typ == content_type:
                cont = i
        model = cont.ising
        adj = [model[(vx-1)%self.x,vy], model[vx,(vy-1)%self.y], model[(vx+1)%self.x,vy], model[vx,(vy+1)%self.y]] #隣接するエージェント
        
        # ising weight test
        #adj = [model[(vx-1)%self.x,vy]*self.users[(vx-1)%self.x,vy].weight, model[vx,(vy-1)%self.y]*self.users[vx,(vy-1)%self.y].weight, model[(vx+1)%self.x,vy]*self.users[(vx+1)%self.x,vy].weight, model[vx,(vy+1)%self.y]*self.users[vx,(vy+1)%self.y].weight]
        
        dE = self.energy(adj, self.j, model[vx,vy])
        T = self.temp(adj)
        P = self.transition_rate(np.absolute(dE), T)

        if model[vx,vy] == 1:
            ix = self.post_cont[0][0] - content[0]
            postlen = len(self.post_cont)
            if postlen > ix:
                self.post_cont[ix][6] += 1 #いいね数加算
        else:
            if dE < 0:
                model[vx,vy] = 1
                self.retweet_or_quotetweet(step,cont.typ, vx, vy, content)
            else:
                rand = random.random()
                if rand <= P:
                    model[vx,vy] = 1
                    self.retweet_or_quotetweet(step,cont.typ, vx, vy, content) 

    #エネルギー
    def energy(self, adj, j, user):
        E = 0 #判断を変えない場合のエネルギー
        Echange = 0 #判断を変えた場合のエネルギー
        for i in adj:
            E = E + (-1)*j*user*i
            Echange = Echange + j*user*i
        dE = Echange - E #判断を変えた場合と変えない場合のエネルギーの差
        return dE

    #温度
    def temp(self, adj):
        s = 0
        for i in adj:
            s = s + i
        s = np.absolute(s)
        T = float(1)/(1+2*s)
        return T
    #温度なし
    def nor_temp(self, adj):
        T = 1
        return T

    #状態遷移確率
    def transition_rate(self, dE, T):
        P = float(np.exp((float(-1)*dE/T)))/2
        return P
    
    #リツートか同じ内容で新たにツイートするか
    def retweet_or_quotetweet(self, step, typ, vx, vy, content):
        which = random.choice(["retweet","quotetweet"])
        if which == "retweet":
            ix = self.post_cont[0][0] - content[0]
            postlen = len(self.post_cont)
            if postlen > ix:
                self.post_cont[ix][5] += 1 #リツイート数加算
                self.post_cont[ix][1] = step
        else:
            Typ = int(random.normalvariate(typ,2)%500)
            cont_typs = [i.typ for i in self.contents] #すでに投稿された記事のタイプのリスト
            if Typ not in cont_typs:
                self.add_contents(Typ,self.x,self.y,vx,vy)
            self.add_post_cont(step,Typ,self.users[vx,vy])
    
    #状態遷移試行
    def conds_trandition(self, visitor):
        vx = visitor.x
        vy = visitor.y
        for i in self.contents:
            model = i.ising
            adj = [model[(vx-1)%self.x,vy], model[vx,(vy-1)%self.y], model[(vx+1)%self.x,vy], model[vx,(vy+1)%self.y]] #隣接するエージェント
            
            # ising weight test
            #adj = [model[(vx-1)%self.x,vy]*self.users[(vx-1)%self.x,vy].weight, model[vx,(vy-1)%self.y]*self.users[vx,(vy-1)%self.y].weight, model[(vx+1)%self.x,vy]*self.users[(vx+1)%self.x,vy].weight, model[vx,(vy+1)%self.y]*self.users[vx,(vy+1)%self.y].weight]
            
            dE = self.energy(adj, self.j, model[vx,vy])
            T = self.temp(adj)
            P = self.transition_rate(np.absolute(dE), T)
            if model[vx,vy] == 1:
                if dE < 0:
                    model[vx,vy] = -1
                else:
                    rand = random.random()
                    if rand <= P:
                        model[vx,vy] = -1
    
    #エンゲージメント率
    def enganement(self, cont, ret, fav):
        eng = 0
        if cont[7] != 0:
            eng = float(cont[5]*ret + cont[6]*fav) / cont[7]
        return eng
    
    #見る記事の数
    def see(self, visitor):
        currentId = 0
        if self.post_cont != []:
            currentId = self.post_cont[0][0]
        unsee = currentId - visitor.already_see #未読記事の総数
        see_rate = 0.57 #記事をスクロールする確率
        see_cont_num = int(unsee * see_rate) #見る記事の数
        return see_cont_num


    #ーーーーーSNS側がアルゴリズム作成に利用できるデーターーーーー

    #閲覧履歴に追加
    def add_history(self, see_cont, visitor):
        if self.post_cont != []:
            delpost = self.post_cont[len(self.post_cont)-1][0]
            visitor.history = [i for i in visitor.history if i[0] > delpost]
        visitor.history = see_cont + visitor.history
    
    #詳細表示に追加
    def add_click(self, visitor, cont):
        delpost = self.post_cont[len(self.post_cont)-1][0]
        visitor.click = [i for i in visitor.click if i[0] > delpost]
        visitor.click = [cont] + visitor.click

    #ユーザー追加
    def add_users(self, x, y):
        self.users[x, y] = User(x, y)
    
    #記事情報追加
    def add_contents(self, typ, field_x, field_y, post_x, post_y):
        cont = Content(typ,field_x,field_y)
        cont.ising[post_x,post_y] = 1
        self.contents.append(cont)
    
    #投稿された記事を順番にリストに追加
    def add_post_cont(self, time, typ, user):
        len_post_cont = len(self.post_cont)
        Id = len_post_cont
        if len_post_cont != 0:
            Id = self.post_cont[0][0] + 1
        self.post_cont = [[Id,time,typ,user.x,user.y,0,0,0]] + self.post_cont
        if len_post_cont >= 500:
            self.post_cont.pop()


    #ーーーーーユーザーしか知り得ないデーターーーーー
    
    #標準偏差に追加
    def add_std(self, see_cont):
        if len(see_cont) >= 50:
            cont_typs = [cont[2] for cont in see_cont]
            mean = sum(cont_typs)/len(cont_typs)
            squared_diff = [(typ-mean)**2 for typ in cont_typs]
            variance = sum(squared_diff)/len(squared_diff)
            std = variance**0.5
            self.stds.append(std)
    
    #ヒストグラムに追加
    def add_hist(self, see_cont):
        if len(see_cont) >= 50:
            cont_typs = [cont[2] for cont in see_cont]
            hist, bins = np.histogram(cont_typs, bins=np.arange(0,500,10))
            relative_hist = hist/sum(hist)
            self.hists.append([relative_hist, bins])

    #ステップ
    def step(self,step,n_step,series, production):
        rx = random.randint(0,self.x-1)
        ry = random.randint(0,self.y-1)
        visitor = self.users[rx,ry] #ユーザーを1人ランダムで決定
        self.conds_trandition(visitor) #ユーザーの記事に対する注目度の変化を測定
        which = random.choice(["read","tweet"])
        if which == "read": #記事閲覧
            sort = self.select_SERIES(series,visitor,step)
            self.look(sort, visitor, step, n_step, production)
            #visitor.already_see = len(self.post_cont)
            if self.post_cont != []:
                visitor.already_see = self.post_cont[0][0]
        else: #記事投稿
            vtyp = visitor.typ
            typ = int(random.normalvariate(vtyp,5)%500)     
            cont_typs = [i.typ for i in self.contents] #すでに投稿された記事のタイプのリスト
            if typ not in cont_typs:
                self.add_contents(typ,self.x,self.y,visitor.x,visitor.y)
            self.add_post_cont(step,typ,visitor)
        for cont in self.contents:
            self.cond_stock(cont,step)

    #開始
    def start(self,empty_step,n_step):
        for x in range(self.x):
            for y in range(self.y):
                self.add_users(x,y)

                # ising weight test
                self.users[x,y].weight = random.uniform(0,2)
                typ = random.randint(1,500)
                self.users[x,y].typ = typ
        es = round(empty_step/2)
        for i in range(es):
            self.step(i,n_step+empty_step,"TIME_SERIES",False)
            print(i)
        for i in range(es):
            self.step(i+es,n_step+empty_step,self.series,False)
            print(i+es)
        for i in range(n_step):
            self.step(i+empty_step,n_step+empty_step,self.series,True)
            print(i+empty_step)
    

    #ーーーーー記事の情勢の管理ーーーーー
    
    #情勢
    def condition(self,cont):
        m = np.sum(cont.ising)
        M = float(m)/(self.x*self.y)
        return(M)
    
    #情勢をストック
    def cond_stock(self,cont,step):
        M = self.condition(cont)
        cont.cond.append([step,M])
    
    #時間軸にプロット
    def timeplot(self,step,cond):
        plt.ion()
        plt.plot(step,cond,'o')
        plt.draw()
        plt.pause(0.000001)
    
    #プロット終了
    def plotend(self):
        plt.ioff()
        plt.show()



#ーーーーーーーーーーーーーーー可視化ーーーーーーーーーーーーーーー
def std_plot(std_data,file_name):
    X = np.linspace(1,len(std_data),len(std_data))
    Y = std_data

    data = go.Scatter(
        x=X,
        y=Y,
        mode="lines",
        name="standard dev"
    )
    layout = go.Layout(
        xaxis=dict(
            title="x"
        ),
        yaxis=dict(
            title="std"
        )
    )

    fig = dict(data = [data], layout=layout)
    plotly.offline.plot(fig, filename='{}.html'.format(file_name))



def histogram_plot(hist_data,file_name):
    X = [] #階級
    Y = [] #閲覧順
    Z = [] #度数分布
    current = 1
    for hist in hist_data:
        X.extend(np.delete([i-(int(hist[1][1]/2)) for i in hist[1]],0))
        Y.extend([current]*len(hist[0]))
        Z.extend(hist[0])
        current += 1
    
    data = go.Heatmap(
        x=X,
        y=Y,
        z=Z,
        zmin=0,
        zmax=1,
        colorscale="Viridis"
    )
    layout = go.Layout(
        xaxis=dict(
            title="bin"
        ),
        yaxis=dict(
            title="see"
        )
    )

    fig = go.Figure(data=[data], layout=layout)
    plotly.offline.plot(fig, filename='{}.html'.format(file_name))



def condition_plot(contents_data,file_name,prestep,n_step):
    X = [] #time
    Y = [] #contents
    Z = [] #condition
    current = 1
    for cont in contents_data:
        x = np.array(cont.cond)[::30][:,0]
        y = [current]*len(x)
        z = np.array(cont.cond)[::30][:,1]
        X.extend(x)
        Y.extend(y)
        Z.extend(z)
        current += 1
    
    data = go.Scatter3d(
        x=X,
        y=Y,
        z=Z,
        mode="markers",
        marker=dict(
            size=1,
            color=Z,
            colorscale='Viridis',
            showscale=True,
            opacity=0.8
        )
    )
    layout = go.Layout(
        scene=dict(
            xaxis = dict(
                range=[prestep,prestep+n_step],
                title="Time",),
            yaxis = dict(
                range=[0,len(contents_data)],
                title="Contents",),
            zaxis = dict(
                range=[-1,1],
                title="Condition",),
        ),
    
    )
    
    fig = go.Figure(data=[data], layout=layout)
    plotly.offline.plot(fig, filename='{}.html'.format(file_name))


#ーーーーーーーーーーーーーーー実装ーーーーーーーーーーーーーーー
EMPTY_STEP = 2000 #空ステップ数
STEP_NUMBER = 3000 #ステップ数
XY = 20 #人数　XY*XY人

algorithm = 'HISTORY_SERIES' #アルゴリズム

file_name1 = 'PLOT-CONDITION-' + algorithm #ファイル名
file_name2 = 'PLOT-FILTERBUBBLE-' + algorithm
file_name3 = 'PLOT-HISTOGRAM-' + algorithm


newsfeed = NewsFeed(XY,XY,algorithm)
newsfeed.start(EMPTY_STEP,STEP_NUMBER)
print('fin')

condition_plot(newsfeed.contents,file_name1,EMPTY_STEP,STEP_NUMBER)
std_plot(newsfeed.stds,file_name2)
histogram_plot(newsfeed.hists,file_name3)