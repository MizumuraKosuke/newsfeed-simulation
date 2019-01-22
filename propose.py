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
        self.fav = []
        self.ret = []



class Content:
    def __init__(self,typ,x,y):
        self.typ = typ #投稿の種類
        self.ising = np.full((x,y), -1) #イジングモデル
        self.cond = [] #時間ごとの記事の情勢を保持



class NewsFeed:
    def __init__(self,x,y,series,rate):
        self.x = x #ユーザーエージェントフィールドのx軸
        self.y = y #ユーザーエージェントフィールドのy軸
        self.users = np.zeros((self.x,self.y), dtype=object) #ユーザーフィールドの枠組み
        self.series = series
        self.type_num = 100 #タイプの数
        self.contents = [] #コンテンツのタイプごとにオブジェクトを作成し、格納
        self.post_cont = [] #投稿された記事全てを格納。 id, 時間, タイプ, ユーザーx座標, y座標, リツイート数, いいね数, 閲覧数
        self.j = 0.047 #人間関係の強度
        self.stds = [] #ニュースフィードの記事のタイプの標準偏差
        self.hists = [] #ニュースフィードの記事のタイプの度数分布
        self.discom = [] #ニュースフィードの記事のタイプの不快度
        self.typestds = [] #ユーザータイプの標準偏差
        self.rate = rate #提案法の混合率
    
    #アルゴリズム選択関数
    def select_SERIES(self,series,visitor,step):
        if series == "TIME_SERIES":
            return self.TIME_SERIES()
        elif series == "ENGAGE_SERIES":
            return self.ENGAGE_SERIES(visitor, step)
        elif series == "HISTORY_SERIES":
            return self.HISTORY_SERIES(visitor, step)
        elif series == 'PROPOSE_SERIES':
            return self.PROPOSE_SERIES(visitor, step)
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
        score_list = [[sort[i][0], scoreli[i]+self.enganement(sort[i],5,1)] for i in range(sort_len)]
        score_list.sort(key=operator.itemgetter(1),reverse=True)
        engage_list = [self.post_cont[self.post_cont[0][0]-i[0]] for i in score_list]
        return engage_list
    
    #閲覧履歴に基づくアルゴリズム
    def HISTORY_SERIES(self,visitor,step):
        sort = self.TIME_SERIES()
        sort_len = len(sort)
        scoreli = np.linspace(0.02,0,len(sort))
        clicks = visitor.click[:40]
        rets = visitor.ret
        favs = visitor.fav
        for i in range(sort_len):
            for post in clicks:
                if post[2] == sort[i][2]:
                    scoreli[i] += 0.01
            for postId in rets:
                if self.post_cont[len(self.post_cont)-1][0] < postId:
                    post = self.post_cont[self.post_cont[0][0]-postId]
                    if post[2] == sort[i][2]:
                        scoreli[i] += 0.02
            for postId in favs:
                if self.post_cont[len(self.post_cont)-1][0] < postId:
                    post = self.post_cont[self.post_cont[0][0]-postId]
                    if post[2] == sort[i][2]:
                        scoreli[i] += 0.02
        score_list = [[sort[i][0], scoreli[i]] for i in range(sort_len)]
        score_list.sort(key=operator.itemgetter(1),reverse=True)
        history_list = [self.post_cont[self.post_cont[0][0]-i[0]] for i in score_list]
        return history_list
    
    #提案法アルゴリズム
    def PROPOSE_SERIES(self, visitor, step):
        sort = self.HISTORY_SERIES(visitor, step)
        sort_len = len(sort)
        typ_li = [sort[i][2] for i in range(sort_len)]
        top_typs, c = zip(*collections.Counter(typ_li).most_common())
        top_typs = list(top_typs)
        top_typs = [i for i in top_typs if abs(i-sort[0][2]) > 50][:30]
        push_rate = self.rate
        push_number = int(30*push_rate)
        if push_number > len(top_typs):
            push_number = len(top_typs)
        #ins = 9
        #for typ in top_typs:
        #    for i in range(sort_len):
        #        if sort[i][2] == typ:
        #            sort[ins], sort[i] = sort[i], sort[ins]
        #            ins += 10
        #            break
        poplist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
        poplist = random.sample(poplist,push_number)
        for j in range(push_number):
            typ = top_typs[j]
            rand = poplist[j]
            for i in range(sort_len):
                if sort[i][2] == typ:
                    sort[rand], sort[i] = sort[i], sort[rand]
                    break       
        return sort


    #記事閲覧
    def look(self, sort, visitor, step, n_step, production):
        see_cont_num = self.see(visitor)
        see_cont = sort[:see_cont_num]
        if len(see_cont) >= 30:
            hist=self.hist(see_cont)
            self.move_type(visitor, hist)
            if production:
                self.add_std(see_cont)
                self.add_hist(hist)
                #self.add_discomfort(see_cont, visitor)
        self.add_history(see_cont,visitor)
        rrlook = np.linspace(1,0,len(see_cont)) #閲覧確率を上から順に下げていくリスト
        rrtyp = np.array([(1-0.01*abs(visitor.typ-i[2])) for i in see_cont])
        read_rate_list = rrlook * rrtyp
        count = 0
        dis_cont = []
        for cont in see_cont:
            ix = self.post_cont[0][0] - cont[0]
            postlen = len(self.post_cont)
            if postlen > ix:
                self.post_cont[ix][7] += 1 #閲覧数加算
            if random.random() < read_rate_list[count]:
                self.add_click(visitor, cont)
                self.trans(cont,visitor,step)
                dis_cont.append(cont)
            count += 1
        if production:
            self.add_discomfort(dis_cont, visitor)
    
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
        dE = self.energy(adj, self.j, model[vx,vy])
        T = self.temp(adj)
        P = self.transition_rate(np.absolute(dE), T)

        if model[vx,vy] == 1:
            ix = self.post_cont[0][0] - content[0]
            postlen = len(self.post_cont)
            if postlen > ix:
                self.post_cont[ix][6] += 1 #いいね数加算
                self.add_fav(visitor,self.post_cont[ix][0])
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
        Typ = int(random.normalvariate(typ,0.5)%self.type_num)
        ix = self.post_cont[0][0] - content[0]
        postlen = len(self.post_cont)
        if postlen > ix:
            self.post_cont[ix][5] += 1 #リツイート数加算
            self.add_ret(self.users[vx,vy],self.post_cont[ix][0])
        self.add_post_cont(step,Typ,self.users[vx,vy])
    
    #状態遷移試行
    def conds_trandition(self, visitor):
        vx = visitor.x
        vy = visitor.y
        for i in self.contents:
            model = i.ising
            adj = [model[(vx-1)%self.x,vy], model[vx,(vy-1)%self.y], model[(vx+1)%self.x,vy], model[vx,(vy+1)%self.y]] #隣接するエージェント
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
        #print('unsee : {},see cont num: {} current ID :{}, already see: {}'.format(unsee, see_cont_num,currentId, visitor.already_see))
        if see_cont_num > 30:
            see_cont_num = 30
        return see_cont_num
    
    #タイプの遷移
    def move_type(self, visitor, hist):
        vtype = visitor.typ
        current = 0
        for i in hist[0]:
            if i > 0.1:
                cont_typ = int((hist[1][current]+hist[1][current+1])/2)
                diff = cont_typ - vtype
                if diff == 0:
                  visitor.typ = vtype
                elif diff > 0:
                  visitor.typ = vtype + 2
                else:
                  visitor.typ = vtype - 2
            current += 1


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
    def add_contents(self, typ):
        cont = Content(typ, self.x, self.y)
        self.contents.append(cont)
    
    #投稿された記事を順番にリストに追加
    def add_post_cont(self, time, typ, user):
        len_post_cont = len(self.post_cont)
        Id = 0
        if len_post_cont != 0:
            Id = self.post_cont[0][0] + 1
        self.post_cont = [[Id,time,typ,user.x,user.y,0,0,0]] + self.post_cont
        if len_post_cont >= 1000:
            self.post_cont.pop()

    def add_fav(self, visitor, postId):
        delpost = self.post_cont[len(self.post_cont)-1][0]
        visitor.fav = [i for i in visitor.fav if i > delpost]
        visitor.fav.append(postId)
    
    def add_ret(self, visitor, postId):
        delpost = self.post_cont[len(self.post_cont)-1][0]
        visitor.ret = [i for i in visitor.ret if i > delpost]
        visitor.ret.append(postId)


    #ーーーーーユーザーしか知り得ないデーターーーーー
    
    #標準偏差に追加
    def add_std(self, see_cont):
        cont_typs = [cont[2] for cont in see_cont[:50]]
        mean = sum(cont_typs)/len(cont_typs)
        squared_diff = [(typ-mean)**2 for typ in cont_typs]
        variance = sum(squared_diff)/len(squared_diff)
        std = variance**0.5
        self.stds.append(std)
    
    #タイプの標準偏差に追加
    def add_typestd(self):
        typeli = []
        for i in self.users:
            for j in i:
                typeli.append(j.typ)
        mean = sum(typeli)/len(typeli)
        squared_diff = [(typ-mean)**2 for typ in typeli]
        variance = sum(squared_diff)/len(squared_diff)
        std = variance**0.5
        self.typestds.append(std)

    def hist(self, see_cont):
        cont_typs = [cont[2] for cont in see_cont[:30]]
        hist, bins = np.histogram(cont_typs, bins=np.arange(0,self.type_num,3))
        relative_hist = hist/sum(hist)
        return [relative_hist, bins]

    #ヒストグラムに追加
    def add_hist(self, hist):
        self.hists.append(hist)
    
    #不快指数に追加
    def add_discomfort(self, dis_cont, visitor):
        if dis_cont != []:
            cont_typs = [cont[2] for cont in dis_cont[:30]]
            disli = [abs(i-visitor.typ) for i in cont_typs]
            disave = sum(disli)/len(disli)
            self.discom.append(disave)

    #ステップ
    def step(self,step,n_step,series, production):
        rx = random.randint(0,self.x-1)
        ry = random.randint(0,self.y-1)
        visitor = self.users[rx,ry] #ユーザーを1人ランダムで決定
        self.conds_trandition(visitor) #ユーザーの記事に対する注目度の変化を測定
        if production:
            self.add_typestd()
        which = random.random()
        if which <= 0.8: #記事閲覧
            sort = self.select_SERIES(series,visitor,step)
            self.look(sort, visitor, step, n_step, production)
            if self.post_cont != []:
                visitor.already_see = self.post_cont[0][0]
        else: #記事投稿
            vtyp = visitor.typ
            typ = int(random.normalvariate(vtyp,0.5)%self.type_num)
            self.add_post_cont(step,typ,visitor)
            for i in self.contents:
                if i.typ == typ:
                    i.ising[visitor.x, visitor.y] = 1
        for cont in self.contents:
            self.cond_stock(cont,step)

    #開始
    def start(self,empty_step,n_step):
        for i in range(self.type_num):
            self.add_contents(i)
        #typelin = np.linspace(0,1,self.x)
        for x in range(self.x):
            for y in range(self.y):
                self.add_users(x,y)
                self.users[x,y].typ = random.randint(0,self.type_num-1)
                #self.users[x,y].typ = int(self.type_num*typelin[x]*typelin[y])%self.type_num
                #if random.random() >= 0.5:
                #    self.users[x,y].typ = int(random.normalvariate(10,0.5)%self.type_num)
                #else:
                #    self.users[x,y].typ = int(random.normalvariate(70,0.5)%self.type_num)
        es = round(empty_step/2)
        for i in range(es):
            self.step(i,n_step+empty_step,"TIME_SERIES",False)
            #print(i)
        print('time fin')
        for i in range(es):
            self.step(i+es,n_step+empty_step,self.series,False)
            #print(i+es)
        print('empty fin')
        for i in range(n_step):
            self.step(i+empty_step,n_step+empty_step,self.series,True)
            #print(i+empty_step)
    

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



def discomfort_plot(discom_data,file_name):
    X = np.linspace(1,len(discom_data),len(discom_data))
    Y = discom_data

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
            title="discomfort"
        )
    )

    fig = dict(data = [data], layout=layout)
    plotly.offline.plot(fig, filename='{}.html'.format(file_name))



def incon_plot(incon_data, file_name, rates):
    X = rates
    Y = incon_data
    time = [19.50] * len(rates)
    engage = [16.03] * len(rates)
    history = [12.51] * len(rates)

    data = go.Scatter(
        x=X,
        y=Y,
        mode="lines",
        name="proposed series"
    )
    T = go.Scatter(
        x=X,
        y=time,
        mode="lines",
        name="time series"
    )
    E = go.Scatter(
        x=X,
        y=engage,
        mode="lines",
        name="engagement series"
    )
    H= go.Scatter(
        x=X,
        y=history,
        mode="lines",
        name="history series"
    )
    layout = go.Layout(
        xaxis=dict(
            title="frequency"
        ),
        yaxis=dict(
            title="inconvenience index"
        )
    )

    fig = dict(data = [data,T,E,H], layout=layout)
    plotly.offline.plot(fig, filename='{}.html'.format(file_name))



def scattered_plot(scattered_data, file_name, rates):
    X = rates
    Y = scattered_data
    time = [18.95] * len(rates)
    engage = [3.94] * len(rates)
    history = [7.01] * len(rates)

    data = go.Scatter(
        x=X,
        y=Y,
        mode="lines",
        name="proposed series"
    )
    T = go.Scatter(
        x=X,
        y=time,
        mode="lines",
        name="time series"
    )
    E = go.Scatter(
        x=X,
        y=engage,
        mode="lines",
        name="engage series"
    )
    H= go.Scatter(
        x=X,
        y=history,
        mode="lines",
        name="history series"
    )
    layout = go.Layout(
        xaxis=dict(
            title="frequency"
        ),
        yaxis=dict(
            title="scattered"
        )
    )

    fig = dict(data = [data,T,E,H], layout=layout)
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



def simulation(EMPTY_STEP, STEP_NUMBER, XY, rates, SIM_NUM):
    stdsum = [[] for i in range(len(rates))]
    dissum = [[] for i in range(len(rates))]
    current = 0
    for rate in rates:
        for i in range(SIM_NUM):
            file_name1 = 'PLOT-CONDITION-' + 'PROPOSE_SERIES' #ファイル名
            file_name2 = 'PLOT-FILTERBUBBLE-' + 'PROPOSE_SERIES'
            file_name3 = 'PLOT-HISTOGRAM-' + 'PROPOSE_SERIES'
            file_name4 = 'PLOT-DISCOMFORT-' + 'PROPOSE_SERIES'
            
            print('PROPOSE_SERIES rate: {} times: {}'.format(rate,i+1))
            newsfeed = NewsFeed(XY,XY,'PROPOSE_SERIES',rate)
            newsfeed.start(EMPTY_STEP,STEP_NUMBER)
            print('fin')
            
            #condition_plot(newsfeed.contents,file_name1,EMPTY_STEP,STEP_NUMBER)
            #std_plot(newsfeed.stds,file_name2)
            #histogram_plot(newsfeed.hists,file_name3)
            #discomfort_plot(newsfeed.discom,file_name4)

            print('std: {}'.format(sum(newsfeed.stds)/len(newsfeed.stds)))
            print('不快指数: {}'.format(sum(newsfeed.discom)/len(newsfeed.discom)))
            print('\n')

            stdsum[current].append(sum(newsfeed.stds)/len(newsfeed.stds))
            dissum[current].append(sum(newsfeed.discom)/len(newsfeed.discom))
        current += 1
        std = [round(sum(stdsum[i])/len(stdsum[i]), 2) for i in range(len(stdsum)) if stdsum[i] != []]
        inc = [round(sum(dissum[i])/len(dissum[i]), 2) for i in range(len(dissum)) if dissum[i] != []]
        scattered_plot(std, 'PLOT-SCATTERED', rates)
        incon_plot(inc, 'PLOT-INCON', rates)
        print('\n')
    print('\n\n')
    print('--------------結果---------------')
    print('-----標準偏差-----')
    for i in range(len(std)):
        print('{}: {}'.format(rates[i],std[i]))
    print('-----不快度-----')
    for i in range(len(inc)):
        print('{}: {}'.format(rates[i],inc[i]))

#ーーーーーーーーーーーーーーー実装ーーーーーーーーーーーーーーー
EMPTY_STEP = 10000 #空ステップ数
STEP_NUMBER = 4000 #ステップ数
XY = 35 #人数　XY*XY人
current = 0
rates = np.arange(0, 0.6, 0.03)
SIM_NUM = 10
print(rates)
print('\n')
simulation(EMPTY_STEP, STEP_NUMBER, XY, rates, SIM_NUM)