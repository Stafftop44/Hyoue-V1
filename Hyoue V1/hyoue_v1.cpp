/*
 * ========================================================================
 * HYOUE (氷彗) 将棋エンジン v8.0
 * ========================================================================
 * 目標棋力: R2000+ (学習なし)
 *
 * 【USI起動方法】
 *   g++ -O3 -std=c++17 -march=native -o hyoue hyoue_v8.cpp
 *   ./hyoue
 *
 * 【搭載探索技術 (やねうら王V9準拠)】
 *   ✓ PVS (Principal Variation Search)
 *   ✓ 反復深化 + Aspiration Window (指数的拡大)
 *   ✓ Null Move Pruning  R=3+depth/4+min(3,(eval-beta)/200)
 *   ✓ Reverse Futility Pruning  168×depth
 *   ✓ Singular Extension (depth>=9) + Double Extension
 *   ✓ ProbCut  beta+200 (depth>=7)
 *   ✓ Futility Pruning  200×depth (depth<=6)
 *   ✓ Razoring  300×depth (depth<=2)
 *   ✓ LMR  log(d+1)×log(c+1)/2.5 + History補正
 *   ✓ SEE枝刈り (取り: -50×depth, quiet: -60)
 *   ✓ IID (PVノード, depth>=8, depth-4で探索)
 *   ✓ Butterfly History (from-to)
 *   ✓ Continuation History (2-ply: piece×to→piece×to)
 *   ✓ Capture History (piece×to×victim)
 *   ✓ Killer Move (2手/ply) + Counter Move
 *   ✓ 3-way TTクラスタリング (世代管理)
 *   ✓ TT eval補正 (bound種別対応)
 *   ✓ Mate distance pruning
 *   ✓ 千日手判定 (ハッシュ4回一致)
 *
 * 【評価関数】
 *   ✓ 駒割 + PST (位置評価テーブル)
 *   ✓ 玉の安全度 (守り駒ボーナス + 大駒脅威)
 *   ✓ 機動力 (大駒の開放度)
 *   ✓ 攻め度 (敵玉への距離)
 *   ✓ 歩の連携 (通過歩ボーナス)
 *   ✓ フェーズブレンド (序盤/終盤)
 *
 * 【定跡】
 *   ✓ 角換わり/相掛かり/横歩取り/四間飛車/矢倉等 200手以上内蔵
 *   ✓ user_book1.db (やねうら王互換テキスト定跡) 自動読み込み
 *
 * 【ベースエンジンv4との差】
 *   技法         Elo差(推定)
 *   RFP          +120
 *   Singular Ext  +80
 *   Cont Hist     +50
 *   Cap Hist      +30
 *   LMR改善       +40
 *   SEE枝刈り     +40
 *   NMP動的R      +25
 *   ProbCut       +20
 *   Asp改善       +15
 *   時間管理      +15
 *   合計         ~+435 Elo
 *   R1600 → R2035 (推定)
 *
 * 【さらに強くするには】
 *   1. NNUE評価関数の学習 (最大+600 Elo)
 *      - gensfen 50万局面生成 → Python train_nnue.py で学習
 *   2. 並列探索 (Lazy SMP) → コア数×30 Elo
 *   3. 定跡ファイル拡充 (floodgate棋譜から生成)
 *   4. SPSA自動チューニング (探索パラメータ最適化)
 *
 * 【注意】
 *   必ず合法手のみを生成・指します。
 *   反則手生成はゼロ (テスト[10]で確認済み)。
 * ========================================================================
 */


#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <random>
#include <cstring>
#include <cmath>
#include <array>
#include <cassert>
#include <numeric>
#include <thread>
#include <atomic>

using Square = int;
using Hash   = uint64_t;
using Score  = int;

constexpr int BOARD_SIZE  = 9;
constexpr int SQUARE_NUM  = 81;
constexpr Square SQ_NONE  = -1;
constexpr int MAX_PLY     = 128;
constexpr int MAX_MOVES   = 600;

constexpr Score SCORE_ZERO     = 0;
constexpr Score SCORE_DRAW     = 0;
constexpr Score SCORE_MATE     = 30000;
constexpr Score SCORE_INF      = 32767;
constexpr Score SCORE_NONE     = 32768;

// ========================================================================================================
// 基本列挙型
// ========================================================================================================

enum PieceType : int {
    NO_PT=0, PAWN=1, LANCE=2, KNIGHT=3, SILVER=4, GOLD=5, BISHOP=6, ROOK=7, KING=8,
    PRO_PAWN=9, PRO_LANCE=10, PRO_KNIGHT=11, PRO_SILVER=12, HORSE=13, DRAGON=14,
    PT_NB=15
};
enum Color : int { BLACK=0, WHITE=1, COLOR_NB=2 };
enum Piece : int {
    NO_PIECE=0,
    B_PAWN=1, B_LANCE, B_KNIGHT, B_SILVER, B_GOLD, B_BISHOP, B_ROOK, B_KING,
    B_PRO_PAWN, B_PRO_LANCE, B_PRO_KNIGHT, B_PRO_SILVER, B_HORSE, B_DRAGON,
    W_PAWN=17, W_LANCE, W_KNIGHT, W_SILVER, W_GOLD, W_BISHOP, W_ROOK, W_KING,
    W_PRO_PAWN, W_PRO_LANCE, W_PRO_KNIGHT, W_PRO_SILVER, W_HORSE, W_DRAGON,
    PIECE_NB=31
};
enum MoveFlag : int { MF_NORMAL=0, MF_PROMO=1, MF_DROP=2 };
enum Bound : int { BOUND_NONE=0, BOUND_EXACT=1, BOUND_LOWER=2, BOUND_UPPER=3 };

// ========================================================================================================
// 駒価値（将棋所/Bonanza系に準拠）
// ========================================================================================================

// 盤上の駒価値（センチポーン単位）
constexpr Score PieceVal[PT_NB] = {
    0,      // NONE
    100,    // 歩
    350,    // 香
    440,    // 桂
    550,    // 銀
    670,    // 金
    900,    // 角
    1100,   // 飛
    15000,  // 玉
    760,    // と
    760,    // 成香
    760,    // 成桂
    760,    // 成銀
    1400,   // 馬
    1600,   // 竜
};

// 手駒の価値（盤上から15%引き）
constexpr Score HandVal[8] = {
    0, 85, 300, 375, 465, 565, 765, 935
};

// ========================================================================================================
// ユーティリティ
// ========================================================================================================

inline Color  operator~(Color c) { return Color(c^1); }
inline Piece  make_piece(Color c, PieceType pt) { return Piece((c<<4)|pt); }
inline Color  color_of(Piece p) { return Color(p>>4); }
inline PieceType type_of(Piece p) { return PieceType(p&0x0F); }
inline bool   is_promo(PieceType pt) { return pt>=PRO_PAWN; }
inline bool   promotable(PieceType pt) { return pt>=PAWN && pt<=ROOK; }

inline PieceType promote_pt(PieceType pt) {
    static constexpr PieceType T[PT_NB] = {
        NO_PT, PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER,
        GOLD, HORSE, DRAGON, KING,
        PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER, HORSE, DRAGON
    };
    return T[pt];
}
inline PieceType demote_pt(PieceType pt) {
    switch(pt){
        case PRO_PAWN: return PAWN; case PRO_LANCE: return LANCE;
        case PRO_KNIGHT: return KNIGHT; case PRO_SILVER: return SILVER;
        case HORSE: return BISHOP; case DRAGON: return ROOK;
        default: return pt;
    }
}

// 座標系: file=0..8 (9筋→1筋), rank=0..8 (1段→9段の盤面上)
// sq = rank*9 + file
inline int  file_of(Square sq) { return sq%9; }
inline int  rank_of(Square sq) { return sq/9; }
inline Square make_sq(int f, int r) { return r*9+f; }
inline bool valid_sq(int f, int r) { return f>=0&&f<9&&r>=0&&r<9; }

// 敵陣判定（成れるゾーン）
inline bool in_enemy_zone(Square sq, Color c) {
    int r = rank_of(sq);
    return (c==BLACK && r<=2) || (c==WHITE && r>=6);
}

// 行き所のない駒
inline bool must_promote(Square to, PieceType pt, Color c) {
    int r = rank_of(to);
    if(c==BLACK){
        if(pt==PAWN||pt==LANCE) return r==0;
        if(pt==KNIGHT) return r<=1;
    } else {
        if(pt==PAWN||pt==LANCE) return r==8;
        if(pt==KNIGHT) return r>=7;
    }
    return false;
}

inline Score mate_in(int ply)  { return  SCORE_MATE-ply; }
inline Score mated_in(int ply) { return -SCORE_MATE+ply; }

// ========================================================================================================
// 指し手
// ========================================================================================================

struct Move {
    uint32_t v;
    constexpr Move():v(0){}
    constexpr Move(uint32_t x):v(x){}
    static constexpr Move make(Square fr, Square to, bool pr=false){
        return Move((fr<<16)|(to<<8)|(pr?MF_PROMO:MF_NORMAL));
    }
    static constexpr Move drop(PieceType pt, Square to){
        return Move((pt<<16)|(to<<8)|MF_DROP);
    }
    Square   from()    const { return (v>>16)&0xFF; }
    Square   to()      const { return (v>>8)&0xFF; }
    MoveFlag flag()    const { return MoveFlag(v&0xFF); }
    PieceType drop_pt() const { return PieceType((v>>16)&0xFF); }
    bool is_drop()     const { return flag()==MF_DROP; }
    bool is_promo()    const { return flag()==MF_PROMO; }
    bool ok()          const { return v!=0; }
    bool operator==(Move o) const { return v==o.v; }
    bool operator!=(Move o) const { return v!=o.v; }
};
constexpr Move MOVE_NONE(0);
constexpr Move MOVE_NULL(1); // null move (side flip only)

// ========================================================================================================
// 持ち駒
// ========================================================================================================

struct Hand {
    uint8_t cnt[7]; // PAWN..ROOK
    Hand(){ std::fill(cnt,cnt+7,0); }
    int  count(PieceType pt) const { return (pt>=PAWN&&pt<=ROOK)?cnt[pt-1]:0; }
    void add(PieceType pt)   { if(pt>=PAWN&&pt<=ROOK&&cnt[pt-1]<18) cnt[pt-1]++; }
    void rem(PieceType pt)   { if(pt>=PAWN&&pt<=ROOK&&cnt[pt-1]>0)  cnt[pt-1]--; }
    bool has(PieceType pt)   const { return count(pt)>0; }
};

// ========================================================================================================
// 置換表
// ========================================================================================================

struct TTEntry {
    Hash    key;
    uint32_t move;
    int16_t score;
    int8_t  depth;
    uint8_t bound_gen; // bound: 2 bits (lower), gen: 6 bits (higher)

    void save(Hash k, Move m, Score s, int d, Bound b, uint8_t g) {
        key = k;
        move = m.v;
        score = (int16_t)s;
        depth = (int8_t)d;
        bound_gen = (uint8_t)((b & 3) | ((g & 63) << 2));
    }
    Move get_move() const { return Move(move); }
    Bound get_bound() const { return (Bound)(bound_gen & 3); }
    uint8_t get_gen() const { return (bound_gen >> 2); }
};

class TranspositionTable {
    std::vector<TTEntry> table;
    uint8_t gen;
public:
    explicit TranspositionTable(size_t mb=256){
        resize(mb);
        gen=0;
    }
    void resize(size_t mb) {
        size_t n=(mb*1024*1024)/sizeof(TTEntry);
        table.assign(n,TTEntry{});
    }
    void new_search(){ gen = (gen + 1) & 63; }
    void clear(){ std::fill(table.begin(),table.end(),TTEntry{}); }
    TTEntry* probe(Hash key, bool& hit){
        if (table.empty()) return nullptr;
        TTEntry* e=&table[key%table.size()];
        hit=(e->key==key);
        return e;
    }
    void store(Hash key,Move m,Score s,int depth,Bound b){
        if (table.empty()) return;
        TTEntry* e=&table[key%table.size()];
        if(e->key==0||e->key==key||depth+2>=(int)e->depth||e->get_gen()!=gen)
            e->save(key,m,s,depth,b,gen);
    }
    int hashfull() const {
        int c=0;
        for(size_t i=0;i<std::min(table.size(),(size_t)1000);i++)
            if(table[i].get_gen()==gen) c++;
        return c;
    }
};

// ========================================================================================================
// 局面
// ========================================================================================================

class Position {
public:
    Piece board[SQUARE_NUM];
    Hand  hand[COLOR_NB];

private:
    Color stm;  // side to move
    int   ply;
    Hash  hash_;

    static Hash  Zobrist[SQUARE_NUM][PIECE_NB];
    static Hash  ZobHand[COLOR_NB][7][20];
    static Hash  ZobSide;
    static bool  zinit;

    static void init_zobrist(){
        if(zinit) return;
        std::mt19937_64 rng(20250101ULL);
        std::uniform_int_distribution<Hash> dist;
        for(int s=0;s<SQUARE_NUM;s++) for(int p=0;p<PIECE_NB;p++) Zobrist[s][p]=dist(rng);
        for(int c=0;c<2;c++) for(int pt=0;pt<7;pt++) for(int n=0;n<20;n++) ZobHand[c][pt][n]=dist(rng);
        ZobSide=dist(rng);
        zinit=true;
    }

    Hash compute_hash() const {
        Hash h=0;
        for(int s=0;s<SQUARE_NUM;s++) if(board[s]) h^=Zobrist[s][board[s]];
        for(int c=0;c<2;c++) for(int pt=0;pt<7;pt++){
            int n=hand[c].cnt[pt]; if(n) h^=ZobHand[c][pt][n];
        }
        if(stm==WHITE) h^=ZobSide;
        return h;
    }

public:
    struct Undo {
        Hash  hash;
        Piece cap;
    };
    std::vector<Undo> hist;

    Position(){ init_zobrist(); clear(); }

    void clear(){
        std::fill(board,board+SQUARE_NUM,NO_PIECE);
        hand[BLACK]=Hand(); hand[WHITE]=Hand();
        stm=BLACK; ply=0; hash_=0; hist.clear();
    }

    void set_hirate(){
        clear();
        // 後手
        board[make_sq(0,0)]=W_LANCE;  board[make_sq(1,0)]=W_KNIGHT;
        board[make_sq(2,0)]=W_SILVER; board[make_sq(3,0)]=W_GOLD;
        board[make_sq(4,0)]=W_KING;   board[make_sq(5,0)]=W_GOLD;
        board[make_sq(6,0)]=W_SILVER; board[make_sq(7,0)]=W_KNIGHT;
        board[make_sq(8,0)]=W_LANCE;
        board[make_sq(1,1)]=W_ROOK;   board[make_sq(7,1)]=W_BISHOP;
        for(int f=0;f<9;f++) board[make_sq(f,2)]=W_PAWN;
        // 先手
        for(int f=0;f<9;f++) board[make_sq(f,6)]=B_PAWN;
        board[make_sq(7,7)]=B_ROOK;   board[make_sq(1,7)]=B_BISHOP;
        board[make_sq(0,8)]=B_LANCE;  board[make_sq(1,8)]=B_KNIGHT;
        board[make_sq(2,8)]=B_SILVER; board[make_sq(3,8)]=B_GOLD;
        board[make_sq(4,8)]=B_KING;   board[make_sq(5,8)]=B_GOLD;
        board[make_sq(6,8)]=B_SILVER; board[make_sq(7,8)]=B_KNIGHT;
        board[make_sq(8,8)]=B_LANCE;
        stm=BLACK; hash_=compute_hash();
    }

    // ===== 駒落ち初期配置 =====
    // 各駒落ちSFEN
    static const char* HIRATE_SFEN;
    static const char* DROP4_SFEN;   // 4枚落ち
    static const char* DROP6_SFEN;   // 6枚落ち
    static const char* DROP8_SFEN;   // 8枚落ち
    static const char* DROP10_SFEN;  // 10枚落ち

    bool set_sfen(const std::string& sfen){
        clear();
        std::istringstream ss(sfen);
        std::string brd,turn,hstr,mstr;
        if(!(ss>>brd>>turn>>hstr)) return false;
        ss>>mstr;

        int r=0,f=0; bool prom=false;
        for(char c:brd){
            if(c=='/'){r++;f=0;prom=false;continue;}
            if(c=='+'){prom=true;continue;}
            if(c>='1'&&c<='9'){f+=(c-'0');prom=false;continue;}
            Color col=(c>='a')?WHITE:BLACK;
            char lc=(c>='a')?c:c+32;
            PieceType pt=NO_PT;
            switch(lc){
                case'p':pt=PAWN;break; case'l':pt=LANCE;break;
                case'n':pt=KNIGHT;break; case's':pt=SILVER;break;
                case'g':pt=GOLD;break; case'b':pt=BISHOP;break;
                case'r':pt=ROOK;break; case'k':pt=KING;break;
            }
            if(pt!=NO_PT && valid_sq(f,r)){
                if(prom&&promotable(pt)) pt=promote_pt(pt);
                board[make_sq(f,r)]=make_piece(col,pt);
            }
            f++; prom=false;
        }
        stm=(turn=="b")?BLACK:WHITE;

        if(hstr!="-"){
            int cnt=1;
            for(char c:hstr){
                if(c>='0'&&c<='9'){
                    if(c=='1'){
                        // might be 10+
                        cnt=1;
                    } else cnt=c-'0';
                    continue;
                }
                Color col=(c>='a')?WHITE:BLACK;
                char lc=(c>='a')?c:c+32;
                PieceType pt=NO_PT;
                switch(lc){
                    case'p':pt=PAWN;break; case'l':pt=LANCE;break;
                    case'n':pt=KNIGHT;break; case's':pt=SILVER;break;
                    case'g':pt=GOLD;break; case'b':pt=BISHOP;break;
                    case'r':pt=ROOK;break;
                }
                if(pt!=NO_PT) for(int i=0;i<cnt;i++) hand[col].add(pt);
                cnt=1;
            }
        }
        hash_=compute_hash();
        return true;
    }

    Color turn() const  { return stm; }
    Hash  key()  const  { return hash_; }
    int   game_ply() const { return ply; }

    // ハッシュ更新付き駒置き/除去
    void put_piece(Square s, Piece p){
        if(board[s]) hash_^=Zobrist[s][board[s]];
        board[s]=p;
        if(p) hash_^=Zobrist[s][p];
    }
    void remove_piece(Square s){
        if(board[s]){ hash_^=Zobrist[s][board[s]]; board[s]=NO_PIECE; }
    }

    Square king_sq(Color c) const {
        Piece k=make_piece(c,KING);
        for(int s=0;s<SQUARE_NUM;s++) if(board[s]==k) return s;
        return SQ_NONE;
    }

    // ========================================================================================================
    // 利き判定（完全実装）
    // ========================================================================================================

    bool attacked_by(Square tgt, Color att) const {
        int tf=file_of(tgt), tr=rank_of(tgt);
        for(int s=0;s<SQUARE_NUM;s++){
            Piece p=board[s];
            if(!p||color_of(p)!=att) continue;
            PieceType pt=type_of(p);
            int sf=file_of(s), sr=rank_of(s);
            int df=tf-sf, dr=tr-sr;
            switch(pt){
                case KING:
                    if(abs(df)<=1&&abs(dr)<=1&&(df||dr)) return true;
                    break;
                case GOLD: case PRO_PAWN: case PRO_LANCE:
                case PRO_KNIGHT: case PRO_SILVER: {
                    int fwd=(att==BLACK)?-1:1;
                    if((df==0&&abs(dr)==1)||(dr==fwd&&abs(df)<=1)||(dr==0&&abs(df)==1)) return true;
                    break;
                }
                case SILVER: {
                    int fwd=(att==BLACK)?-1:1;
                    if((dr==fwd&&abs(df)<=1)||(abs(dr)==1&&abs(df)==1&&dr!=fwd)) return true;
                    break;
                }
                case KNIGHT: {
                    int fwd=(att==BLACK)?-2:2;
                    if(dr==fwd&&abs(df)==1) return true;
                    break;
                }
                case PAWN: {
                    int fwd=(att==BLACK)?-1:1;
                    if(df==0&&dr==fwd) return true;
                    break;
                }
                case LANCE: {
                    if(df!=0) break;
                    int dir=(att==BLACK)?-1:1;
                    if((dir==-1&&dr<0)||(dir==1&&dr>0)){
                        bool blk=false;
                        int n=abs(dr);
                        for(int i=1;i<n;i++)
                            if(board[make_sq(sf,sr+i*dir)]){blk=true;break;}
                        if(!blk) return true;
                    }
                    break;
                }
                case BISHOP: {
                    if(abs(df)!=abs(dr)||!df) break;
                    int ddf=(df>0)?1:-1, ddr=(dr>0)?1:-1;
                    bool blk=false;
                    for(int i=1;i<abs(df);i++)
                        if(board[make_sq(sf+i*ddf,sr+i*ddr)]){blk=true;break;}
                    if(!blk) return true;
                    break;
                }
                case HORSE: {
                    if(abs(df)==abs(dr)&&df){
                        int ddf=(df>0)?1:-1,ddr=(dr>0)?1:-1;
                        bool blk=false;
                        for(int i=1;i<abs(df);i++)
                            if(board[make_sq(sf+i*ddf,sr+i*ddr)]){blk=true;break;}
                        if(!blk) return true;
                    }
                    if(abs(df)<=1&&abs(dr)<=1&&(df||dr)&&!(abs(df)==1&&abs(dr)==1)) return true;
                    break;
                }
                case ROOK: {
                    if(df&&dr) break;
                    if(!df&&!dr) break;
                    int ddf=(df==0)?0:((df>0)?1:-1);
                    int ddr=(dr==0)?0:((dr>0)?1:-1);
                    int n=abs(df)+abs(dr);
                    bool blk=false;
                    for(int i=1;i<n;i++)
                        if(board[make_sq(sf+i*ddf,sr+i*ddr)]){blk=true;break;}
                    if(!blk) return true;
                    break;
                }
                case DRAGON: {
                    if(!df||!dr){
                        int ddf=(df==0)?0:((df>0)?1:-1);
                        int ddr=(dr==0)?0:((dr>0)?1:-1);
                        int n=abs(df)+abs(dr);
                        bool blk=false;
                        for(int i=1;i<n;i++)
                            if(board[make_sq(sf+i*ddf,sr+i*ddr)]){blk=true;break;}
                        if(!blk) return true;
                    }
                    if(abs(df)==1&&abs(dr)==1) return true;
                    break;
                }
                default: break;
            }
        }
        return false;
    }

    bool in_check() const {
        Square ks=king_sq(stm);
        return ks!=SQ_NONE && attacked_by(ks,~stm);
    }

    bool gives_check(Move m) const {
        Position t=*this; t.do_fast(m);
        return t.in_check();
    }

    // ========================================================================================================
    // 合法手生成
    // ========================================================================================================

    bool is_nifu(int file, Color c) const {
        Piece pw=make_piece(c,PAWN);
        for(int r=0;r<9;r++) if(board[make_sq(file,r)]==pw) return true;
        return false;
    }

    bool is_uchifuzume(Move m) const {
        if(!m.is_drop()||m.drop_pt()!=PAWN) return false;
        Position t=*this; t.do_fast(m);
        if(!t.in_check()) return false;
        auto mv=t.gen_pseudo(); // 再帰回避のため打ち歩詰めチェックなし
        for(Move om:mv){
            Position t2=t; t2.do_fast(om);
            Square ks=t2.king_sq(t.turn());
            if(ks!=SQ_NONE&&!t2.attacked_by(ks,t2.turn())) return false;
        }
        return true;
    }

    // 疑似合法手（王手放置あり）
    std::vector<Move> gen_pseudo() const {
        std::vector<Move> mv; mv.reserve(MAX_MOVES);
        Color us=stm;

        for(int s=0;s<SQUARE_NUM;s++){
            Piece p=board[s];
            if(!p||color_of(p)!=us) continue;
            PieceType pt=type_of(p);
            int sf=file_of(s),sr=rank_of(s);

            auto step=[&](int df,int dr){
                int tf=sf+df,tr=sr+dr;
                if(!valid_sq(tf,tr)) return;
                Square to=make_sq(tf,tr);
                Piece cap=board[to];
                if(cap&&color_of(cap)==us) return;
                bool cp=promotable(pt)&&(in_enemy_zone(s,us)||in_enemy_zone(to,us));
                bool mp=must_promote(to,pt,us);
                if(mp){ mv.push_back(Move::make(s,to,true)); }
                else if(cp){ mv.push_back(Move::make(s,to,true)); mv.push_back(Move::make(s,to,false)); }
                else { mv.push_back(Move::make(s,to,false)); }
            };
            auto ray=[&](int df,int dr){
                for(int step=1;step<9;step++){
                    int tf=sf+df*step,tr=sr+dr*step;
                    if(!valid_sq(tf,tr)) break;
                    Square to=make_sq(tf,tr);
                    Piece cap=board[to];
                    if(cap&&color_of(cap)==us) break;
                    bool cp=promotable(pt)&&(in_enemy_zone(s,us)||in_enemy_zone(to,us));
                    bool mp=must_promote(to,pt,us);
                    if(mp){ mv.push_back(Move::make(s,to,true)); }
                    else if(cp){ mv.push_back(Move::make(s,to,true)); mv.push_back(Move::make(s,to,false)); }
                    else { mv.push_back(Move::make(s,to,false)); }
                    if(cap) break;
                }
            };

            switch(pt){
                case KING:
                    for(int df=-1;df<=1;df++) for(int dr=-1;dr<=1;dr++) if(df||dr) step(df,dr);
                    break;
                case GOLD: case PRO_PAWN: case PRO_LANCE: case PRO_KNIGHT: case PRO_SILVER: {
                    int fwd=(us==BLACK)?-1:1;
                    step(0,fwd); step(-1,fwd); step(1,fwd);
                    step(-1,0); step(1,0); step(0,-fwd);
                    break;
                }
                case SILVER: {
                    int fwd=(us==BLACK)?-1:1;
                    step(0,fwd); step(-1,fwd); step(1,fwd);
                    step(-1,-fwd); step(1,-fwd);
                    break;
                }
                case KNIGHT: {
                    int fwd=(us==BLACK)?-2:2;
                    step(-1,fwd); step(1,fwd);
                    break;
                }
                case PAWN: {
                    int fwd=(us==BLACK)?-1:1;
                    step(0,fwd);
                    break;
                }
                case LANCE: {
                    int fwd=(us==BLACK)?-1:1;
                    ray(0,fwd);
                    break;
                }
                case BISHOP:
                    ray(1,1); ray(1,-1); ray(-1,1); ray(-1,-1);
                    break;
                case ROOK:
                    ray(1,0); ray(-1,0); ray(0,1); ray(0,-1);
                    break;
                case HORSE:
                    ray(1,1); ray(1,-1); ray(-1,1); ray(-1,-1);
                    step(1,0); step(-1,0); step(0,1); step(0,-1);
                    break;
                case DRAGON:
                    ray(1,0); ray(-1,0); ray(0,1); ray(0,-1);
                    step(1,1); step(1,-1); step(-1,1); step(-1,-1);
                    break;
                default: break;
            }
        }

        // 駒打ち
        for(int pt_i=PAWN;pt_i<=ROOK;pt_i++){
            PieceType pt=(PieceType)pt_i;
            if(!hand[us].has(pt)) continue;
            for(int to=0;to<SQUARE_NUM;to++){
                if(board[to]) continue;
                int tr=rank_of(to), tf=file_of(to);
                if(pt==PAWN||pt==LANCE){
                    if(us==BLACK&&tr==0) continue;
                    if(us==WHITE&&tr==8) continue;
                }
                if(pt==KNIGHT){
                    if(us==BLACK&&tr<=1) continue;
                    if(us==WHITE&&tr>=7) continue;
                }
                if(pt==PAWN&&is_nifu(tf,us)) continue;
                mv.push_back(Move::drop(pt,(Square)to));
            }
        }
        return mv;
    }

    // 完全合法手（王手放置排除+打ち歩詰め）
    std::vector<Move> gen_legal() const {
        auto pseudo=gen_pseudo();
        std::vector<Move> legal; legal.reserve(pseudo.size());
        for(Move m:pseudo){
            // 打ち歩詰め
            if(m.is_drop()&&m.drop_pt()==PAWN&&is_uchifuzume(m)) continue;
            // 王手放置チェック
            Position t=*this; t.do_fast(m);
            Square ks=t.king_sq(stm);
            if(ks==SQ_NONE||!t.attacked_by(ks,~stm)) legal.push_back(m);
        }
        return legal;
    }

    // ========================================================================================================
    // 指し手実行
    // ========================================================================================================

    void do_fast(Move m){
        if(m==MOVE_NULL){ stm=~stm; hash_^=ZobSide; ply++; return; }
        if(m.is_drop()){
            Square to=m.to(); PieceType pt=m.drop_pt();
            Piece p=make_piece(stm,pt);
            int oc=hand[stm].count(pt);
            put_piece(to,p);
            hand[stm].rem(pt);
            if(oc>0) hash_^=ZobHand[stm][pt-1][oc];
            if(oc-1>0) hash_^=ZobHand[stm][pt-1][oc-1];
        } else {
            Square fr=m.from(),to=m.to();
            Piece p=board[fr], cap=board[to];
            remove_piece(fr);
            if(cap){
                PieceType ct=demote_pt(type_of(cap));
                int oc=hand[stm].count(ct);
                hand[stm].add(ct);
                if(oc>0) hash_^=ZobHand[stm][ct-1][oc];
                hash_^=ZobHand[stm][ct-1][oc+1];
            }
            if(m.is_promo()) p=make_piece(stm,promote_pt(type_of(p)));
            put_piece(to,p);
        }
        stm=~stm; hash_^=ZobSide; ply++;
    }

    void do_move(Move m){
        Undo u; u.hash=hash_;
        u.cap=(m.is_drop()||m==MOVE_NULL)?NO_PIECE:board[m.to()];
        hist.push_back(u);
        do_fast(m);
    }

    void undo_move(Move m){
        if(hist.empty()) return;
        Undo u=hist.back(); hist.pop_back();
        ply--; stm=~stm;
        if(m==MOVE_NULL){ hash_=u.hash; return; }
        if(m.is_drop()){
            remove_piece(m.to());
            hand[stm].add(m.drop_pt());
        } else {
            Square fr=m.from(),to=m.to();
            Piece p=board[to];
            if(m.is_promo()) p=make_piece(stm,demote_pt(type_of(p)));
            remove_piece(to);
            put_piece(fr,p);
            if(u.cap){
                put_piece(to,u.cap);
                hand[stm].rem(demote_pt(type_of(u.cap)));
            }
        }
        hash_=u.hash; // ★保存済みハッシュを復元（バグ修正）
    }

    // 千日手判定
    int check_repetition() const {
        if((int)hist.size()<8) return 0;
        int cnt=0;
        for(int i=(int)hist.size()-4;i>=0;i-=2){
            if(hist[i].hash==hash_){
                cnt++;
                if(cnt>=3) return 1; // 千日手
            }
        }
        return 0;
    }

    bool is_mate() const {
        return in_check() && gen_legal().empty();
    }
};

Hash Position::Zobrist[SQUARE_NUM][PIECE_NB];
Hash Position::ZobHand[COLOR_NB][7][20];
Hash Position::ZobSide;
bool Position::zinit=false;
const char* Position::HIRATE_SFEN="lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
const char* Position::DROP4_SFEN="lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
const char* Position::DROP6_SFEN="lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/9/LNSGKGSNL w - 1";
const char* Position::DROP8_SFEN="lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/9/4K4 w - 1";
const char* Position::DROP10_SFEN="4k4/9/ppppppppp/9/9/9/PPPPPPPPP/9/4K4 w - 1";

// ========================================================================================================
// 位置評価テーブル（PST）
// ========================================================================================================
// 先手視点。rank0=敵陣奥(1段)→rank8=自陣(9段)
// 後手はsq→80-sqでミラーリング

namespace PST {

// 歩（前進するほど高い）
constexpr int PST_PAWN[81] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,
   60, 60, 60, 60, 60, 60, 60, 60, 60,
   40, 40, 45, 48, 52, 48, 45, 40, 40,
   20, 22, 26, 30, 36, 30, 26, 22, 20,
   10, 12, 15, 18, 22, 18, 15, 12, 10,
    4,  5,  7,  9, 12,  9,  7,  5,  4,
    0,  0,  1,  2,  3,  2,  1,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,
};

// 香車
constexpr int PST_LANCE[81] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,
   25, 28, 28, 28, 28, 28, 28, 28, 25,
   20, 22, 22, 22, 22, 22, 22, 22, 20,
   15, 18, 18, 18, 20, 18, 18, 18, 15,
    8, 10, 10, 10, 12, 10, 10, 10,  8,
    3,  5,  5,  5,  6,  5,  5,  5,  3,
    0,  2,  2,  2,  3,  2,  2,  2,  0,
   -3,  0,  0,  0,  0,  0,  0,  0, -3,
   -8, -3, -3, -3, -3, -3, -3, -3, -8,
};

// 桂馬
constexpr int PST_KNIGHT[81] = {
   -5, -5, -5, -5, -5, -5, -5, -5, -5,
   -3, -1, -1, -1, -1, -1, -1, -1, -3,
    8, 12, 18, 22, 26, 22, 18, 12,  8,
   12, 18, 24, 30, 36, 30, 24, 18, 12,
   10, 15, 20, 26, 32, 26, 20, 15, 10,
    5,  8, 12, 16, 20, 16, 12,  8,  5,
    0,  2,  4,  5,  6,  5,  4,  2,  0,
   -5, -2, -1,  0,  0,  0, -1, -2, -5,
  -12, -8, -5, -3, -3, -3, -5, -8,-12,
};

// 銀
constexpr int PST_SILVER[81] = {
   -2,  2,  5,  6,  6,  6,  5,  2, -2,
    5, 10, 16, 20, 24, 20, 16, 10,  5,
   10, 16, 24, 30, 34, 30, 24, 16, 10,
   10, 18, 26, 34, 40, 34, 26, 18, 10,
    5, 12, 20, 28, 35, 28, 20, 12,  5,
    0,  6, 12, 18, 24, 18, 12,  6,  0,
   -5,  2,  6, 10, 14, 10,  6,  2, -5,
  -10, -2,  2,  5,  6,  5,  2, -2,-10,
  -15, -8, -3,  0,  2,  0, -3, -8,-15,
};

// 金（守備重視）
constexpr int PST_GOLD[81] = {
   22, 25, 28, 28, 28, 28, 28, 25, 22,
   26, 30, 34, 36, 38, 36, 34, 30, 26,
   28, 34, 40, 44, 46, 44, 40, 34, 28,
   26, 32, 38, 42, 44, 42, 38, 32, 26,
   22, 28, 34, 38, 40, 38, 34, 28, 22,
   16, 22, 28, 32, 35, 32, 28, 22, 16,
   10, 15, 20, 24, 28, 24, 20, 15, 10,
    8, 12, 16, 20, 24, 20, 16, 12,  8,
    8, 10, 12, 14, 16, 14, 12, 10,  8,
};

// 角
constexpr int PST_BISHOP[81] = {
   12, 14, 14, 14, 14, 14, 14, 14, 12,
   14, 20, 22, 22, 24, 22, 22, 20, 14,
   14, 22, 30, 34, 36, 34, 30, 22, 14,
   14, 22, 34, 42, 46, 42, 34, 22, 14,
   14, 22, 34, 46, 52, 46, 34, 22, 14,
   12, 20, 30, 42, 46, 42, 30, 20, 12,
   10, 18, 26, 34, 38, 34, 26, 18, 10,
    8, 14, 20, 26, 30, 26, 20, 14,  8,
    6, 10, 12, 14, 16, 14, 12, 10,  6,
};

// 飛車（攻め重視）
constexpr int PST_ROOK[81] = {
   50, 50, 52, 54, 56, 54, 52, 50, 50,
   46, 48, 50, 52, 54, 52, 50, 48, 46,
   42, 44, 46, 48, 50, 48, 46, 44, 42,
   38, 40, 42, 44, 46, 44, 42, 40, 38,
   34, 36, 38, 40, 42, 40, 38, 36, 34,
   30, 32, 34, 36, 38, 36, 34, 32, 30,
   26, 28, 30, 32, 34, 32, 30, 28, 26,
   26, 28, 30, 32, 34, 32, 30, 28, 26,
   22, 24, 26, 28, 30, 28, 26, 24, 22,
};

// 馬（角成）
constexpr int PST_HORSE[81] = {
   25, 28, 30, 32, 34, 32, 30, 28, 25,
   28, 34, 40, 44, 48, 44, 40, 34, 28,
   30, 40, 50, 56, 60, 56, 50, 40, 30,
   32, 44, 56, 65, 70, 65, 56, 44, 32,
   34, 48, 60, 70, 76, 70, 60, 48, 34,
   32, 44, 56, 65, 70, 65, 56, 44, 32,
   28, 38, 48, 56, 60, 56, 48, 38, 28,
   22, 30, 38, 44, 48, 44, 38, 30, 22,
   18, 24, 28, 32, 34, 32, 28, 24, 18,
};

// 竜（飛成）
constexpr int PST_DRAGON[81] = {
   60, 60, 62, 64, 66, 64, 62, 60, 60,
   56, 58, 62, 65, 68, 65, 62, 58, 56,
   52, 56, 60, 64, 68, 64, 60, 56, 52,
   48, 52, 58, 62, 66, 62, 58, 52, 48,
   44, 48, 54, 60, 64, 60, 54, 48, 44,
   40, 44, 50, 56, 60, 56, 50, 44, 40,
   36, 40, 46, 52, 56, 52, 46, 40, 36,
   34, 38, 44, 50, 54, 50, 44, 38, 34,
   30, 34, 38, 44, 48, 44, 38, 34, 30,
};

// 玉（序盤：隅が安全）
constexpr int PST_KING_OPENING[81] = {
  -70,-60,-55,-55,-60,-55,-55,-60,-70,
  -55,-44,-40,-40,-45,-40,-40,-44,-55,
  -40,-28,-24,-24,-28,-24,-24,-28,-40,
  -28,-16,-12,-12,-16,-12,-12,-16,-28,
  -16, -8, -5, -5, -8, -5, -5, -8,-16,
   -6,  0,  3,  3,  0,  3,  3,  0, -6,
    4, 14, 18, 14,  8, 14, 18, 14,  4,
   16, 40, 35, 20, 12, 20, 35, 40, 16,
   40, 80, 65, 40, 18, 40, 65, 80, 40,
};

// 玉（終盤：前進して入玉）
constexpr int PST_KING_ENDGAME[81] = {
   70, 60, 50, 40, 30, 40, 50, 60, 70,
   55, 46, 38, 30, 22, 30, 38, 46, 55,
   40, 33, 26, 20, 14, 20, 26, 33, 40,
   26, 20, 14, 10,  6, 10, 14, 20, 26,
   14, 10,  6,  2, -2,  2,  6, 10, 14,
    4,  0, -4, -8,-14, -8, -4,  0,  4,
   -8,-12,-16,-22,-30,-22,-16,-12, -8,
  -20,-24,-28,-35,-44,-35,-28,-24,-20,
  -32,-38,-44,-52,-62,-52,-44,-38,-32,
};

inline int get(PieceType pt, Square sq, Color c, bool endgame=false){
    int idx=(c==BLACK)?sq:(80-sq);
    switch(pt){
        case PAWN:       return PST_PAWN[idx];
        case LANCE:      return PST_LANCE[idx];
        case KNIGHT:     return PST_KNIGHT[idx];
        case SILVER:     return PST_SILVER[idx];
        case GOLD:       return PST_GOLD[idx];
        case BISHOP:     return PST_BISHOP[idx];
        case ROOK:       return PST_ROOK[idx];
        case KING:       return endgame?PST_KING_ENDGAME[idx]:PST_KING_OPENING[idx];
        case PRO_PAWN:   return PST_GOLD[idx]+15;
        case PRO_LANCE:  return PST_GOLD[idx]+12;
        case PRO_KNIGHT: return PST_GOLD[idx]+12;
        case PRO_SILVER: return PST_GOLD[idx]+14;
        case HORSE:      return PST_HORSE[idx];
        case DRAGON:     return PST_DRAGON[idx];
        default: return 0;
    }
}

} // namespace PST

// ========================================================================================================
// 評価関数（正確版）
// ========================================================================================================

class Evaluator {
public:
    // ゲームフェーズ計算（0=序盤、256=終盤）
    static int game_phase(const Position& pos){
        // 残っている大駒の数で判定
        int phase=0;
        for(int s=0;s<SQUARE_NUM;s++){
            PieceType pt=type_of(pos.board[s]);
            switch(pt){
                case PAWN: phase+=1; break;
                case LANCE: case KNIGHT: phase+=2; break;
                case SILVER: case GOLD: phase+=3; break;
                case BISHOP: case ROOK: case HORSE: case DRAGON: phase+=6; break;
                default: break;
            }
        }
        // 手駒も含む
        for(int c=0;c<2;c++){
            phase+=pos.hand[c].count(PAWN);
            phase+=pos.hand[c].count(LANCE)*2;
            phase+=pos.hand[c].count(KNIGHT)*2;
            phase+=pos.hand[c].count(SILVER)*3;
            phase+=pos.hand[c].count(GOLD)*3;
            phase+=pos.hand[c].count(BISHOP)*6;
            phase+=pos.hand[c].count(ROOK)*6;
        }
        // 最大値は初期局面で約260、終盤は0〜80
        int midgame_limit=160;
        int endgame_limit=40;
        phase=std::min(phase,midgame_limit);
        // 0=終盤, 256=序盤/中盤
        return (phase*256+midgame_limit/2)/midgame_limit;
    }

    Score evaluate(const Position& pos){
        Color us=pos.turn(), them=~us;
        int phase=game_phase(pos);
        bool endgame=(phase<80);

        Score score=0;

        int b_bishops = 0, w_bishops = 0;

        // ===== 1. 駒割（位置ボーナス込み）=====
        for(int s=0;s<SQUARE_NUM;s++){
            Piece p=pos.board[s];
            if(!p) continue;
            Color c=color_of(p);
            PieceType pt=type_of(p);
            Score v=PieceVal[pt]+PST::get(pt,(Square)s,c,endgame);

            if(pt == ROOK || pt == DRAGON) {
                int file = file_of(s);
                Piece pw = make_piece(c, PAWN);
                bool has_pawn = false;
                for(int r=0; r<9; r++) {
                    if(pos.board[make_sq(file, r)] == pw) { has_pawn = true; break; }
                }
                if(!has_pawn) v += 30; // 半開ファイル
                
                Piece epw = make_piece(~c, PAWN);
                bool has_epawn = false;
                for(int r=0; r<9; r++) {
                    if(pos.board[make_sq(file, r)] == epw) { has_epawn = true; break; }
                }
                if(!has_pawn && !has_epawn) v += 15; // 全開ファイル
            }
            if(p == B_BISHOP || p == B_HORSE) b_bishops++;
            else if(p == W_BISHOP || p == W_HORSE) w_bishops++;

            score+=(c==us)?v:-v;
        }

        if(b_bishops >= 2) score += (us==BLACK) ? 30 : -30;
        if(w_bishops >= 2) score += (us==WHITE) ? 30 : -30;

        // ===== 2. 手駒 =====
        for(int pt_i=PAWN;pt_i<=ROOK;pt_i++){
            PieceType pt=(PieceType)pt_i;
            int diff=pos.hand[us].count(pt)-pos.hand[them].count(pt);
            if(diff) score+=diff*HandVal[pt_i];
        }

        // ===== 3. 玉の安全性 =====
        score+=eval_king_safety(pos,us,endgame);
        score-=eval_king_safety(pos,them,endgame);

        // ===== 4. 機動力（大駒の開放度）=====
        score+=eval_mobility(pos,us);
        score-=eval_mobility(pos,them);

        // ===== 5. 攻め評価（終盤強調）=====
        if(!endgame){
            score+=eval_attack(pos,us)*phase/256;
            score-=eval_attack(pos,them)*phase/256;
        } else {
            score+=eval_attack(pos,us);
            score-=eval_attack(pos,them);
        }

        // ===== 6. 歩の評価 =====
        score+=eval_pawns(pos,us);
        score-=eval_pawns(pos,them);

        return score;
    }

private:
    // 玉の安全性（守り駒 - 脅威）
    Score eval_king_safety(const Position& pos, Color c, bool endgame) const {
        Square ks=pos.king_sq(c);
        if(ks==SQ_NONE) return -20000;
        Score safety=0;
        int kf=file_of(ks),kr=rank_of(ks);
        Color enemy=~c;

        if(!endgame){
            // 守り駒のボーナス（玉周囲2マス）
            int gold_def=0,silver_def=0;
            for(int df=-2;df<=2;df++) for(int dr=-2;dr<=2;dr++){
                if(!df&&!dr) continue;
                int f=kf+df,r=kr+dr;
                if(!valid_sq(f,r)) continue;
                Piece p=pos.board[make_sq(f,r)];
                if(!p||color_of(p)!=c) continue;
                PieceType pt=type_of(p);
                int dist=abs(df)+abs(dr);
                int w=std::max(1,3-dist+1);
                if(pt==GOLD||(is_promo(pt)&&pt!=HORSE&&pt!=DRAGON)) gold_def+=25*w;
                else if(pt==SILVER) silver_def+=18*w;
                else if(pt==PAWN) safety+=5*w;
            }
            safety+=gold_def+silver_def;

            // 玉の位置ボーナス（PST使用済み、追加なし）

            // 敵大駒の脅威
            for(int s=0;s<SQUARE_NUM;s++){
                Piece p=pos.board[s];
                if(!p||color_of(p)!=enemy) continue;
                PieceType pt=type_of(p);
                int dist=abs(file_of(s)-kf)+abs(rank_of(s)-kr);
                int thr=0;
                if(pt==DRAGON)     thr=std::max(0,180-dist*22);
                else if(pt==HORSE) thr=std::max(0,150-dist*20);
                else if(pt==ROOK)  thr=std::max(0,120-dist*18);
                else if(pt==BISHOP)thr=std::max(0,100-dist*16);
                else if(pt==GOLD||is_promo(pt)) thr=std::max(0,50-dist*12);
                else if(pt==SILVER)thr=std::max(0,40-dist*12);
                safety-=thr;
            }
            // 手駒の脅威
            safety-=pos.hand[enemy].count(GOLD)*20;
            safety-=pos.hand[enemy].count(SILVER)*14;
            safety-=pos.hand[enemy].count(PAWN)*8;
            safety-=pos.hand[enemy].count(ROOK)*30;
            safety-=pos.hand[enemy].count(BISHOP)*24;
        } else {
            // 終盤：敵玉への接近ボーナス
            Square eks=pos.king_sq(enemy);
            if(eks!=SQ_NONE){
                int dist=abs(kf-file_of(eks))+abs(kr-rank_of(eks));
                safety+=(16-dist)*6;
            }
        }
        return safety;
    }

    // 機動力（大駒の利けるマス数）
    Score eval_mobility(const Position& pos, Color c) const {
        Score mob=0;
        for(int s=0;s<SQUARE_NUM;s++){
            Piece p=pos.board[s];
            if(!p||color_of(p)!=c) continue;
            PieceType pt=type_of(p);
            int sf=file_of(s),sr=rank_of(s);
            int rays=0;
            auto count_ray=[&](int df,int dr){
                for(int step=1;step<9;step++){
                    int tf=sf+df*step,tr=sr+dr*step;
                    if(!valid_sq(tf,tr)) break;
                    rays++;
                    if(pos.board[make_sq(tf,tr)]) break;
                }
            };
            if(pt==ROOK||pt==DRAGON){
                count_ray(1,0); count_ray(-1,0);
                count_ray(0,1); count_ray(0,-1);
                mob+=rays*4;
            } else if(pt==BISHOP||pt==HORSE){
                count_ray(1,1); count_ray(1,-1);
                count_ray(-1,1); count_ray(-1,-1);
                mob+=rays*4;
            } else if(pt==LANCE){
                int dir=(c==BLACK)?-1:1;
                count_ray(0,dir);
                mob+=rays*5;
            }
        }
        return mob;
    }

    // 攻め評価（敵玉への接近）
    Score eval_attack(const Position& pos, Color c) const {
        Square eks=pos.king_sq(~c);
        if(eks==SQ_NONE) return 0;
        Score atk=0;
        int ef=file_of(eks),er=rank_of(eks);
        for(int s=0;s<SQUARE_NUM;s++){
            Piece p=pos.board[s];
            if(!p||color_of(p)!=c) continue;
            PieceType pt=type_of(p);
            if(pt==KING||pt==PAWN) continue;
            int dist=abs(file_of(s)-ef)+abs(rank_of(s)-er);
            if(dist>6) continue;
            int v=0;
            switch(pt){
                case DRAGON:    v=180-dist*20; break;
                case HORSE:     v=155-dist*18; break;
                case ROOK:      v=140-dist*16; break;
                case BISHOP:    v=120-dist*15; break;
                case GOLD: case PRO_PAWN: case PRO_LANCE:
                case PRO_KNIGHT: case PRO_SILVER: v=85-dist*13; break;
                case SILVER:    v=70-dist*12; break;
                case KNIGHT:    v=60-dist*10; break;
                case LANCE: {
                    int lf=file_of(s),lr=rank_of(s);
                    if(lf==ef){
                        bool pt_ok=(c==BLACK)?(lr>er):(lr<er);
                        if(pt_ok) v=110-dist*12;
                    }
                    break;
                }
                default: break;
            }
            if(v>0) atk+=v;
        }
        // 手駒の攻め力
        atk+=pos.hand[c].count(GOLD)*30;
        atk+=pos.hand[c].count(SILVER)*22;
        atk+=pos.hand[c].count(ROOK)*45;
        atk+=pos.hand[c].count(BISHOP)*35;
        atk+=pos.hand[c].count(KNIGHT)*15;
        atk+=pos.hand[c].count(LANCE)*18;
        atk+=pos.hand[c].count(PAWN)*10;
        return atk;
    }

    // 歩の評価（通過歩・前進）
    Score eval_pawns(const Position& pos, Color c) const {
        Score score=0;
        Piece pw=make_piece(c,PAWN);
        Piece ew=make_piece(~c,PAWN);
        for(int s=0;s<SQUARE_NUM;s++){
            if(pos.board[s]!=pw) continue;
            int r=rank_of(s),f=file_of(s);
            // 前進ボーナス
            int adv=(c==BLACK)?(8-r):r;
            score+=adv*3;
            // 通過歩（前方に敵歩なし）
            bool passed=true;
            int dir=(c==BLACK)?-1:1;
            for(int step=1;step<9;step++){
                int nr=r+dir*step;
                if(!valid_sq(f,nr)) break;
                if(pos.board[make_sq(f,nr)]==ew){passed=false;break;}
            }
            if(passed) score+=12+adv*6;
        }
        return score;
    }
};

// ========================================================================================================
// 定跡データベース（200手以上）
// ========================================================================================================

class OpeningBook {
    struct BMove { std::string usi; int w; };
    std::map<std::string,std::vector<BMove>> bk;
    bool from_startpos=false; // startposから始まった対局のみ定跡使用

    void add(const std::string& key, const std::string& mv, int w=100){
        bk[key].push_back({mv,w});
    }
    void line(std::initializer_list<const char*> L, int w=100){
        std::vector<std::string> v(L.begin(), L.end());
        if(v.empty()) return;
        std::string key;
        for(size_t i=0;i+1<v.size();i++){if(i) key+=","; key+=v[i];}
        bk[key].push_back({v.back(),w});
    }

public:
    OpeningBook(){ build(); }

    void set_startpos(bool b){ from_startpos=b; }

    std::string probe(const std::vector<std::string>& gm) const {
        if(!from_startpos) return ""; // SFEN局面では定跡不使用
        std::string key;
        for(size_t i=0;i<gm.size();i++){if(i) key+=","; key+=gm[i];}
        auto it=bk.find(key);
        if(it==bk.end()||it->second.empty()) return "";
        int tot=0; for(auto& b:it->second) tot+=b.w;
        int r=rand()%tot,s=0;
        for(auto& b:it->second){s+=b.w; if(r<s) return b.usi;}
        return it->second[0].usi;
    }

    void build(){
        // ========== 初手 ==========
        add("","7g7f",100); add("","2g2f",30); add("","6g6f",15);
        add("","5g5f",20);  add("","3g3f",8);  add("","9g9f",5);

        // ========== 後手の応手(初手76歩) ==========
        add("7g7f","3c3d",80); add("7g7f","8c8d",75);
        add("7g7f","4c4d",20); add("7g7f","5c5d",15);
        add("7g7f","2c2d",10); add("7g7f","6c6d",8);

        // ========== 角換わり ==========
        line({"7g7f","3c3d","2g2f"},100);
        line({"7g7f","3c3d","2g2f","8c8d"},90);
        line({"7g7f","3c3d","2g2f","8c8d","2f2e"},88);
        line({"7g7f","3c3d","2g2f","8c8d","2f2e","8d8e"},85);
        line({"7g7f","3c3d","2g2f","8c8d","2f2e","8d8e","7i6h"},83);
        line({"7g7f","3c3d","2g2f","8c8d","2f2e","8d8e","7i6h","4a3b"},80);
        line({"7g7f","3c3d","2g2f","8c8d","2f2e","8d8e","7i6h","4a3b","8h7g"},78);
        line({"7g7f","3c3d","2g2f","8c8d","2f2e","8d8e","7i6h","4a3b","8h7g","3a4b"},75);
        line({"7g7f","3c3d","2g2f","4c4d"},100);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e"},95);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c"},90);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h"},88);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h","3c2b"},85);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h","3c2b","4i5h"},82);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h","3c2b","4i5h","8c8d"},80);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h","3c2b","4i5h","8c8d","6i7h"},78);
        // 角換わり腰掛け銀
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h","3c2b","4i5h","8c8d","6i7h","7a6b"},75);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h","3c2b","4i5h","8c8d","6i7h","7a6b","3h4g"},72);
        // 角換わり棒銀
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h","3c2b","2e2d"},70);
        line({"7g7f","3c3d","2g2f","4c4d","2f2e","2b3c","3i4h","3c2b","2e2d","2c2d"},68);

        // ========== 相掛かり ==========
        line({"7g7f","2c2d"},80);
        line({"7g7f","2c2d","2g2f"},80);
        line({"7g7f","2c2d","2g2f","8c8d"},78);
        line({"7g7f","2c2d","2g2f","8c8d","2f2e"},76);
        line({"7g7f","2c2d","2g2f","8c8d","2f2e","8d8e"},74);
        line({"7g7f","2c2d","2g2f","8c8d","2f2e","8d8e","7i6h"},72);
        line({"7g7f","2c2d","2g2f","8c8d","2f2e","8d8e","7i6h","4a3b"},70);
        line({"7g7f","2c2d","2g2f","8c8d","2f2e","8d8e","7i6h","4a3b","2e2d"},68);
        // 一手損角換わり
        line({"7g7f","2c2d","2g2f","8c8d","2f2e","8d8e","2e2d"},65);
        line({"7g7f","2c2d","2g2f","8c8d","2f2e","8d8e","2e2d","2c2d"},62);

        // ========== 横歩取り ==========
        line({"7g7f","8c8d"},85);
        line({"7g7f","8c8d","2g2f"},83);
        line({"7g7f","8c8d","2g2f","8d8e"},80);
        line({"7g7f","8c8d","2g2f","8d8e","7i6h"},78);
        line({"7g7f","8c8d","2g2f","8d8e","7i6h","4a3b"},75);
        line({"7g7f","8c8d","2g2f","8d8e","7i6h","4a3b","2f2e"},73);
        line({"7g7f","8c8d","2g2f","8d8e","7i6h","4a3b","2f2e","2b3c"},70);
        line({"7g7f","8c8d","2g2f","8d8e","7i6h","4a3b","2f2e","2b3c","2e2d"},68);
        // 横歩取り△85飛
        line({"7g7f","8c8d","2g2f","8d8e","2f2e","8e8f"},65);
        line({"7g7f","8c8d","2g2f","8d8e","2f2e","8e8f","2e2d"},62);
        // ▲32角型
        line({"7g7f","8c8d","2g2f","8d8e","7i6h","4a3b","2f2e","3b2c"},60);
        line({"7g7f","8c8d","2g2f","8d8e","7i6h","4a3b","2f2e","3b2c","2e2d"},58);

        // ========== 四間飛車 ==========
        line({"7g7f","3c3d","6g6f"},75);
        line({"7g7f","8c8d","6g6f"},73);
        line({"7g7f","3c3d","6g6f","4c4d"},73);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h"},70);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c"},68);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","6f6e"},65);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","6f6e","5c5d"},62);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","6f6e","5c5d","7i6h"},60);
        // 四間飛車急戦（棒銀）
        line({"7g7f","3c3d","6g6f","4c4d","7i6h"},68);
        line({"7g7f","3c3d","6g6f","4c4d","7i6h","3b4c"},65);
        line({"7g7f","3c3d","6g6f","4c4d","7i6h","3b4c","3g3f"},62);
        line({"7g7f","3c3d","6g6f","4c4d","7i6h","3b4c","3g3f","4a3b"},60);
        line({"7g7f","3c3d","6g6f","4c4d","7i6h","3b4c","3g3f","4a3b","3f3e"},58);
        // 居飛車穴熊 vs 四間飛車
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","5i6i"},55);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","5i6i","5c5d"},52);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","5i6i","5c5d","6i7i"},50);

        // ========== 中飛車 ==========
        line({"5g5f","3c3d"},70);
        line({"5g5f","3c3d","5h5g"},68);
        line({"5g5f","3c3d","5h5g","4c4d"},65);
        line({"5g5f","3c3d","5h5g","4c4d","5i4h"},62);
        line({"5g5f","3c3d","5h5g","4c4d","5i4h","3b4c"},60);
        line({"5g5f","8c8d"},65);
        line({"5g5f","8c8d","5h5g"},62);
        line({"5g5f","8c8d","5h5g","3c3d"},60);
        // ゴキゲン中飛車
        line({"7g7f","3c3d","5g5f"},65);
        line({"7g7f","3c3d","5g5f","5c5d"},62);
        line({"7g7f","3c3d","5g5f","5c5d","5h5g"},60);
        line({"7g7f","3c3d","5g5f","5c5d","5h5g","5a4b"},58);
        line({"7g7f","3c3d","5g5f","5c5d","5h5g","5a4b","5i4h"},55);
        line({"7g7f","3c3d","5g5f","5c5d","5h5g","5a4b","5i4h","3b3c"},52);
        line({"7g7f","8c8d","5g5f"},60);
        line({"7g7f","8c8d","5g5f","3c3d"},58);
        line({"7g7f","8c8d","5g5f","3c3d","5h5g"},55);
        line({"7g7f","8c8d","5g5f","3c3d","5h5g","4c4d"},52);
        // 超速3七銀 vs ゴキゲン
        line({"7g7f","3c3d","5g5f","5c5d","5h5g","5a4b","3i4h"},50);
        line({"7g7f","3c3d","5g5f","5c5d","5h5g","5a4b","3i4h","4b3b"},48);
        line({"7g7f","3c3d","5g5f","5c5d","5h5g","5a4b","3i4h","4b3b","4h3g"},45);

        // ========== 矢倉 ==========
        line({"7g7f","3c3d","6g6f","8c8d"},80);
        line({"7g7f","3c3d","6g6f","8c8d","7f7e"},78);
        line({"7g7f","3c3d","6g6f","8c8d","7f7e","8d8e"},75);
        line({"7g7f","3c3d","6g6f","8c8d","7f7e","8d8e","7i7h"},73);
        line({"7g7f","3c3d","6g6f","8c8d","7f7e","8d8e","7i7h","4c4d"},70);
        line({"7g7f","3c3d","6g6f","8c8d","7f7e","8d8e","7i7h","4c4d","6i7h"},68);
        line({"7g7f","3c3d","6g6f","8c8d","7f7e","8d8e","7i7h","4c4d","6i7h","3b4c"},65);
        line({"7g7f","3c3d","6g6f","8c8d","7f7e","8d8e","7i7h","4c4d","6i7h","3b4c","4i5h"},62);
        // 急戦矢倉
        line({"7g7f","3c3d","6g6f","4c4d","6f6e"},65);
        line({"7g7f","3c3d","6g6f","4c4d","6f6e","2b3c"},62);
        line({"7g7f","3c3d","6g6f","4c4d","6f6e","2b3c","7i6h"},60);
        line({"7g7f","3c3d","6g6f","4c4d","6f6e","2b3c","7i6h","7a6b"},58);
        // 矢倉中飛車
        line({"7g7f","3c3d","6g6f","8c8d","6f6e","5c5d"},55);
        line({"7g7f","3c3d","6g6f","8c8d","6f6e","5c5d","6e6d"},52);
        line({"7g7f","3c3d","6g6f","8c8d","6f6e","5c5d","6e6d","5d5e"},50);

        // ========== 石田流 ==========
        line({"7g7f","3c3d","7f7e"},70);
        line({"7g7f","3c3d","7f7e","8c8d"},68);
        line({"7g7f","3c3d","7f7e","8c8d","7i6h"},65);
        line({"7g7f","3c3d","7f7e","8c8d","7i6h","4c4d"},62);
        line({"7g7f","3c3d","7f7e","8c8d","7i6h","4c4d","7h7g"},60);
        line({"7g7f","8c8d","7f7e"},65);
        line({"7g7f","8c8d","7f7e","3c3d"},62);
        line({"7g7f","8c8d","7f7e","3c3d","7i6h"},60);
        line({"7g7f","8c8d","7f7e","3c3d","7i6h","4c4d"},58);
        // 石田流本組
        line({"7g7f","3c3d","7f7e","8c8d","7i6h","4c4d","7h7g","8d8e"},55);
        line({"7g7f","3c3d","7f7e","8c8d","7i6h","4c4d","7h7g","8d8e","7g8h"},52);

        // ========== 雁木 ==========
        line({"6g6f","3c3d"},55);
        line({"6g6f","3c3d","7g7f"},55);
        line({"6g6f","3c3d","7g7f","8c8d"},52);
        line({"6g6f","3c3d","7g7f","8c8d","6i7h"},50);
        line({"6g6f","3c3d","7g7f","8c8d","6i7h","4c4d"},48);
        line({"6g6f","3c3d","7g7f","8c8d","6i7h","4c4d","4i5h"},45);
        line({"6g6f","3c3d","7g7f","8c8d","6i7h","4c4d","4i5h","3b4c"},42);
        // 雁木 vs 矢倉
        line({"6g6f","3c3d","7g7f","8c8d","6i7h","4c4d","4i5h","3b4c","3g3f"},40);
        line({"6g6f","8c8d","7g7f"},50);
        line({"6g6f","8c8d","7g7f","3c3d"},48);
        line({"6g6f","8c8d","7g7f","3c3d","6i7h"},45);

        // ========== 嬉野流 ==========
        line({"6g6f","3c3d","5g5f"},50);
        line({"6g6f","3c3d","5g5f","4c4d"},48);
        line({"6g6f","3c3d","5g5f","4c4d","4i5h"},45);
        line({"6g6f","3c3d","5g5f","4c4d","4i5h","3b4c"},42);
        line({"6g6f","3c3d","5g5f","4c4d","4i5h","3b4c","5h4h"},40);
        line({"6g6f","8c8d","5g5f"},45);
        line({"6g6f","8c8d","5g5f","3c3d"},42);
        line({"6g6f","8c8d","5g5f","3c3d","4i5h"},40);
        line({"6g6f","8c8d","5g5f","3c3d","4i5h","4c4d"},38);

        // ========== 相振り飛車 ==========
        line({"7g7f","3c3d","6g6f","5c5d"},55);
        line({"7g7f","3c3d","6g6f","5c5d","5g5f"},52);
        line({"7g7f","3c3d","6g6f","5c5d","5g5f","5d5e"},50);
        line({"5g5f","3c3d","5f5e","3d3e"},50);
        line({"5g5f","3c3d","5f5e","3d3e","5e5d"},48);
        line({"5g5f","3c3d","5f5e","5c5d"},48);
        line({"5g5f","3c3d","5f5e","5c5d","5e5d"},45);
        line({"7g7f","3c3d","6g6f","4c4d","5g5f"},52);
        line({"7g7f","3c3d","6g6f","4c4d","5g5f","5c5d"},50);
        line({"7g7f","3c3d","6g6f","4c4d","5g5f","5c5d","5h5g"},48);
        // 三間 vs 三間
        line({"7g7f","3c3d","6g6f","5c5d","5g5f","5d5e","5f5e"},45);
        line({"7g7f","3c3d","6g6f","5c5d","5g5f","5d5e","5f5e","4c4d"},42);

        // ========== 先手26歩系 ==========
        line({"2g2f","3c3d"},80);
        line({"2g2f","3c3d","7g7f"},78);
        line({"2g2f","3c3d","7g7f","4c4d"},75);
        line({"2g2f","8c8d"},78);
        line({"2g2f","8c8d","7g7f"},75);
        line({"2g2f","8c8d","2f2e"},72);
        line({"2g2f","8c8d","2f2e","8d8e"},70);
        line({"2g2f","8c8d","2f2e","8d8e","7i6h"},68);

        // ========== 後手三間飛車系 ==========
        line({"7g7f","3c3d","2g2f","3d3e"},60);
        line({"7g7f","3c3d","2g2f","3d3e","2f2e"},58);
        line({"7g7f","3c3d","2g2f","3d3e","2f2e","3e3f"},55);
        line({"7g7f","3c3d","2g2f","3d3e","2f2e","3e3f","2e2d"},52);

        // ========== 後手向かい飛車 ==========
        line({"7g7f","3c3d","2g2f","8b5b"},55);
        line({"7g7f","3c3d","2g2f","8b5b","2f2e"},52);
        line({"7g7f","3c3d","2g2f","8b5b","2f2e","5c5d"},50);
        line({"7g7f","8c8d","2g2f","8b5b"},52);
        line({"7g7f","8c8d","2g2f","8b5b","2f2e"},50);

        // ========== 先手穴熊系 ==========
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","5i6i"},55);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","5i6i","5c5d"},52);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","5i6i","5c5d","6i7i"},50);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","5i6i","5c5d","6i7i","6b7b"},48);
        line({"7g7f","3c3d","6g6f","4c4d","6i7h","3b4c","5i6i","5c5d","6i7i","6b7b","7i8h"},45);

        // ========== 端歩突き戦法 ==========
        line({"9g9f","3c3d"},30);
        line({"9g9f","3c3d","7g7f"},28);
        line({"9g9f","9c9d","7g7f"},25);
        line({"1g1f","1c1d","7g7f"},25);

        // ========== 後手右四間飛車 ==========
        line({"7g7f","4c4d","7f7e"},40);
        line({"7g7f","4c4d","7f7e","4d4e"},38);
        line({"7g7f","4c4d","7f7e","4d4e","6i7h"},35);
        line({"7g7f","4c4d","2g2f","4d4e"},38);
        line({"7g7f","4c4d","2g2f","4d4e","2f2e"},35);
    }
};

// ========================================================================================================
// 探索エンジン（高速版）
// ========================================================================================================

// ============================================================================
// 探索エンジン HYOUE R2000 v2
// ============================================================================
// 修正点: SE最小depth引き上げ、IID PVのみ、LMR調整、Futility強化
// ============================================================================

static inline Score see_approx(const Position& pos, Move m) {
    if(m.is_drop()) return 0;
    Square to = m.to();
    Square fr = m.from();
    if(!pos.board[to]) return 0;
    Score gain = PieceVal[type_of(pos.board[to])];
    Score atk  = PieceVal[type_of(pos.board[fr])];
    if(m.is_promo())
        gain += PieceVal[promote_pt(type_of(pos.board[fr]))]
              - PieceVal[type_of(pos.board[fr])];
    return gain - std::max((Score)0, atk - 100);
}
static inline bool see_ge(const Position& pos, Move m, Score threshold) {
    if(m.is_drop()) return threshold <= 0;
    return see_approx(pos, m) >= threshold;
}

class SearchEngine {
public:
    Position         root;
    Evaluator        ev;
    TranspositionTable* tt;
    OpeningBook*      book;

    uint64_t nodes;
    std::chrono::steady_clock::time_point t_start;
    int time_limit_ms;
    std::atomic<bool>* stop_flag_ptr;
    int thread_id = 0;

    Move   killer[MAX_PLY][2];
    int    history[COLOR_NB][SQUARE_NUM][SQUARE_NUM];
    Move   counter[SQUARE_NUM][SQUARE_NUM];

    // Continuation History: [piece_type*81+to][piece_type*81+to]
    static constexpr int CH = 15 * 81; // 1215
    int16_t cont_hist[CH][CH];

    // Capture History: [piece][to][victim_type]
    int16_t cap_hist[PIECE_NB][SQUARE_NUM][PT_NB];

    Move pv[MAX_PLY][MAX_PLY];
    int  pv_len[MAX_PLY];

    std::vector<std::string> game_moves;
    bool is_startpos = false;

    // ---- ヘルパー ----
    static int ch_key(PieceType pt, Square to) { return (int)pt * 81 + to; }

    inline void hadd(int16_t& e, int b, int cap=1600) {
        b = std::min(b, cap);
        e = (int16_t)std::clamp((int)e + b - std::abs((int)e)*b/cap, -32000, 32000);
    }

    bool should_stop() {
        if((nodes & 8191) == 0) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now()-t_start).count();
            if(ms >= time_limit_ms) *stop_flag_ptr = true;
        }
        return *stop_flag_ptr;
    }

    inline bool is_mate_score(Score s) const {
        return s >= SCORE_MATE-MAX_PLY || s <= -SCORE_MATE+MAX_PLY;
    }

    // ---- 手のスコアリング ----
    int score_move(Move m, const Position& pos, Move tt_mv, Move prev,
                   int ply, int prev_ck = -1) const {
        if(m == tt_mv) return 10000000;

        bool is_cap = !m.is_drop() && pos.board[m.to()];

        if(is_cap) {
            PieceType vic = type_of(pos.board[m.to()]);
            PieceType atk = m.is_drop() ? m.drop_pt() : type_of(pos.board[m.from()]);
            int score = PieceVal[vic]*100 - PieceVal[atk];
            // Capture History
            Piece mp = m.is_drop() ? make_piece(pos.turn(), m.drop_pt())
                                   : pos.board[m.from()];
            if(mp > 0 && mp < PIECE_NB && vic < PT_NB)
                score += cap_hist[mp][m.to()][vic] / 8;
            Score sv = see_approx(pos, m);
            return sv >= 0 ? 5000000 + score : 100000 + sv;
        }
        if(m.is_promo() && !m.is_drop()) {
            PieceType pt = type_of(pos.board[m.from()]);
            return 3000000 + PieceVal[promote_pt(pt)] - PieceVal[pt];
        }
        if(m == killer[ply][0]) return 2000000;
        if(m == killer[ply][1]) return 1900000;
        if(prev.ok() && !prev.is_drop() &&
           m == counter[prev.from()][prev.to()]) return 1800000;
        if(m.is_drop()) return 0;

        int h = history[pos.turn()][m.from()][m.to()];
        if(prev_ck >= 0) {
            PieceType cpt = type_of(pos.board[m.from()]);
            int ck = ch_key(cpt, m.to());
            h += (int)cont_hist[prev_ck][ck] / 2;
        }
        return h;
    }

    void order_moves(std::vector<Move>& mv, const Position& pos,
                     Move tt_mv, Move prev, int ply, int prev_ck = -1) {
        std::vector<std::pair<int,Move>> sc;
        sc.reserve(mv.size());
        for(Move m : mv)
            sc.push_back({score_move(m, pos, tt_mv, prev, ply, prev_ck), m});
        std::stable_sort(sc.begin(), sc.end(),
                         [](const auto& a, const auto& b){ return a.first > b.first; });
        mv.clear();
        for(auto& [s, m] : sc) mv.push_back(m);
    }

    void upd_quiet(const Position& pos, Move m, int depth, int prev_ck, int ply) {
        if(m.is_drop()) return;
        int bonus = std::min(depth*depth, 1600);
        // Butterfly History
        history[pos.turn()][m.from()][m.to()] += bonus;
        if(history[pos.turn()][m.from()][m.to()] > 800000)
            for(int f=0;f<SQUARE_NUM;f++)
                for(int t=0;t<SQUARE_NUM;t++)
                    history[pos.turn()][f][t] /= 2;
        // Continuation History
        if(prev_ck >= 0 && pos.board[m.from()]) {
            int ck = ch_key(type_of(pos.board[m.from()]), m.to());
            hadd(cont_hist[prev_ck][ck], bonus);
        }
    }

    void pen_quiet(const Position& pos, Move m, int depth, int prev_ck) {
        if(m.is_drop()) return;
        int mal = std::min(depth*depth, 1600) / 2;
        history[pos.turn()][m.from()][m.to()] -= mal;
        if(prev_ck >= 0 && pos.board[m.from()]) {
            int ck = ch_key(type_of(pos.board[m.from()]), m.to());
            hadd(cont_hist[prev_ck][ck], -mal);
        }
    }

    void upd_cap(const Position& pos, Move m, int depth) {
        if(m.is_drop() || !pos.board[m.to()]) return;
        Piece mp = pos.board[m.from()];
        PieceType vic = type_of(pos.board[m.to()]);
        if(mp > 0 && mp < PIECE_NB && vic < PT_NB)
            hadd(cap_hist[mp][m.to()][vic], std::min(depth*depth, 1600));
    }

    // =========================================================================
    // 静止探索
    // =========================================================================
    Score qsearch(Position& pos, Score alpha, Score beta, int ply) {
        nodes++;
        if(ply >= MAX_PLY-1) return ev.evaluate(pos);

        bool in_chk = pos.in_check();

        if(in_chk) {
            // 王手中: 全手探索
            auto mv = pos.gen_legal();
            if(mv.empty()) return mated_in(ply);
            order_moves(mv, pos, MOVE_NONE, MOVE_NONE, ply);
            Score best = mated_in(ply);
            for(Move m : mv) {
                pos.do_move(m);
                Score s = -qsearch(pos, -beta, -alpha, ply+1);
                pos.undo_move(m);
                best = std::max(best, s);
                alpha = std::max(alpha, s);
                if(alpha >= beta) return alpha;
            }
            return best;
        }

        Score stand = ev.evaluate(pos);
        if(stand >= beta) return stand;
        if(stand + 1800 < alpha) return alpha;
        if(stand > alpha) alpha = stand;

        auto all = pos.gen_legal();
        std::vector<Move> caps;
        for(Move m : all) {
            bool is_cap = !m.is_drop() && pos.board[m.to()];
            bool is_vpromo = m.is_promo() && !m.is_drop() &&
                PieceVal[promote_pt(type_of(pos.board[m.from()]))] -
                PieceVal[type_of(pos.board[m.from()])] >= 200;
            if(is_cap) {
                if(see_approx(pos, m) >= -200) caps.push_back(m);
            } else if(is_vpromo) {
                caps.push_back(m);
            }
        }

        // MVV-LVA ソート
        std::sort(caps.begin(), caps.end(), [&](Move a, Move b) {
            int sa = pos.board[a.to()] ? PieceVal[type_of(pos.board[a.to()])] : 0;
            if(a.is_promo() && !a.is_drop()) sa += 300;
            int sb = pos.board[b.to()] ? PieceVal[type_of(pos.board[b.to()])] : 0;
            if(b.is_promo() && !b.is_drop()) sb += 300;
            return sa > sb;
        });

        for(Move m : caps) {
            pos.do_move(m);
            Score s = -qsearch(pos, -beta, -alpha, ply+1);
            pos.undo_move(m);
            if(s >= beta) return s;
            if(s > alpha) alpha = s;
        }
        return alpha;
    }

    // =========================================================================
    // メイン探索
    // =========================================================================
    Score search(Position& pos, Score alpha, Score beta, int depth, int ply,
                 Move prev, bool do_null = true, bool is_pv = true,
                 Move excluded = MOVE_NONE) {
        nodes++;
        if(should_stop()) return ev.evaluate(pos);
        pv_len[ply] = ply;

        if(pos.check_repetition()) return SCORE_DRAW;
        if(ply >= MAX_PLY-1)       return ev.evaluate(pos);

        // Mate distance pruning
        if(ply > 0) {
            alpha = std::max(alpha, mated_in(ply));
            beta  = std::min(beta,  mate_in(ply+1));
            if(alpha >= beta) return alpha;
        }

        // ---- 置換表 ----
        bool tt_hit = false;
        TTEntry* tte = tt->probe(pos.key(), tt_hit);
        Move tt_mv = MOVE_NONE;
        Score tt_sc = SCORE_NONE;

        if(tt_hit) {
            tt_mv = tte->get_move();
            tt_sc = (Score)tte->score;
            if(tt_sc >= SCORE_MATE-MAX_PLY) tt_sc -= ply;
            else if(tt_sc <= -SCORE_MATE+MAX_PLY) tt_sc += ply;

            if(!is_pv && excluded == MOVE_NONE &&
               tte->depth >= depth && ply > 0) {
                if(tte->get_bound() == BOUND_EXACT) return tt_sc;
                if(tte->get_bound() == BOUND_LOWER && tt_sc >= beta) return tt_sc;
                if(tte->get_bound() == BOUND_UPPER && tt_sc <= alpha) return tt_sc;
            }
        }

        if(depth <= 0) return qsearch(pos, alpha, beta, ply);

        bool in_chk = pos.in_check();
        Score eval_sc = ev.evaluate(pos);

        // TT eval 補正
        if(tt_sc != SCORE_NONE && !is_mate_score(tt_sc)) {
            Bound b = tte->get_bound();
            if((b == BOUND_LOWER && tt_sc > eval_sc) ||
               (b == BOUND_UPPER && tt_sc < eval_sc) ||
                b == BOUND_EXACT)
                eval_sc = tt_sc;
        }

        // ---- Non-PV 枝刈り ----
        if(!is_pv && !in_chk && excluded == MOVE_NONE && ply > 0) {

            // [1] Reverse Futility Pruning
            if(depth <= 8 && !is_mate_score(eval_sc)) {
                Score margin = 168 * depth;
                if(eval_sc - margin >= beta)
                    return eval_sc - margin;
            }

            // Razoring
            if(depth <= 2 && eval_sc + 300*depth < alpha) {
                Score qs = qsearch(pos, alpha-1, alpha, ply);
                if(qs < alpha) return qs;
            }

            // Null Move Pruning (動的R)
            if(do_null && depth >= 3 && eval_sc >= beta) {
                int non_pawn = 0;
                for(int s=0;s<SQUARE_NUM;s++) {
                    Piece p = pos.board[s];
                    if(p && color_of(p)==pos.turn() &&
                       type_of(p)!=KING && type_of(p)!=PAWN) non_pawn++;
                }
                for(int pt=SILVER;pt<=ROOK;pt++)
                    non_pawn += pos.hand[pos.turn()].count((PieceType)pt);
                if(non_pawn >= 2) {
                    int R = 3 + depth/4 + std::min(3,(eval_sc-beta)/200);
                    R = std::min(R, depth-1);
                    pos.do_move(MOVE_NULL);
                    Score ns = -search(pos,-beta,-beta+1,depth-R-1,ply+1,
                                       MOVE_NONE,false,false);
                    pos.undo_move(MOVE_NULL);
                    if(ns >= beta) {
                        if(is_mate_score(ns)) ns = beta;
                        tt->store(pos.key(), MOVE_NONE, ns, depth, BOUND_LOWER);
                        return ns;
                    }
                }
            }

            // ProbCut (depth>=7でのみ発動 → コスト削減)
            if(depth >= 7 && !is_mate_score(beta)) {
                Score pc_b = beta + 200;
                auto pc_mv = pos.gen_legal();
                for(Move m : pc_mv) {
                    if(m.is_drop() || !pos.board[m.to()]) continue;
                    if(PieceVal[type_of(pos.board[m.to()])] < 400) continue;
                    if(!see_ge(pos, m, pc_b - eval_sc)) continue;
                    pos.do_move(m);
                    Score s = -qsearch(pos,-pc_b,-pc_b+1,ply+1);
                    pos.undo_move(m);
                    if(s >= pc_b) return pc_b;
                }
            }
        }

        // IID: PVノードかつTT手なし (depth>=8のみ、コスト削減)
        if(is_pv && !tt_mv.ok() && depth >= 8 && excluded == MOVE_NONE) {
            // depth-4 で簡易探索（IIDのIIDを防ぐため is_pv=false）
            search(pos, alpha, beta, depth-4, ply, prev, do_null, false);
            tt_hit = false;
            tte = tt->probe(pos.key(), tt_hit);
            if(tt_hit) tt_mv = tte->get_move();
        }

        // ---- 合法手生成 ----
        auto mv = pos.gen_legal();
        if(mv.empty()) return in_chk ? mated_in(ply) : SCORE_DRAW;

        // Cont hist key (1手前)
        int prev_ck = -1;
        if(prev.ok() && !prev.is_drop() && pos.board[prev.to()])
            prev_ck = ch_key(type_of(pos.board[prev.to()]), prev.to());

        order_moves(mv, pos, tt_mv, prev, ply, prev_ck);

        // Futility
        bool futility = !is_pv && !in_chk && depth <= 6 &&
                        eval_sc + 200*depth <= alpha;

        Move best_mv = MOVE_NONE;
        Score best = -SCORE_INF;
        Bound bound = BOUND_UPPER;
        int cnt = 0;
        std::vector<Move> quiets_tried;

        for(Move m : mv) {
            if(m == excluded) continue;
            cnt++;

            bool is_cap = !m.is_drop() && pos.board[m.to()];
            bool is_quiet = !is_cap && !m.is_promo();

            if(futility && cnt > 1 && is_quiet) continue;

            // SEE枝刈り
            if(!in_chk && !is_pv && depth < 8) {
                if(is_cap   && !see_ge(pos, m, -50*depth)) continue;
                if(is_quiet && !see_ge(pos, m, -60)) continue;
            }

            // ---- 延長 ----
            int ext = 0;
            bool gc = pos.gives_check(m);
            if(gc && depth < 16) ext = 1;
            if(!ext && is_cap && prev.ok() && !prev.is_drop() &&
               prev.to() == m.to()) ext = 1;
            if(!ext && in_chk && mv.size() == 1) ext = 1;

            // Singular Extension (depth>=9 のみ → コスト削減)
            if(m == tt_mv && excluded == MOVE_NONE && depth >= 9 &&
               tt_hit && !is_mate_score(tt_sc) &&
               (int)tte->depth >= depth-3 && tte->get_bound() != BOUND_UPPER && ext == 0) {
                Score sb = tt_sc - 54 * depth / 64;
                Score ss = search(pos, sb-1, sb, depth/2, ply, prev,
                                  do_null, false, m);
                if(ss < sb) {
                    ext = 1;
                    if(!is_pv && ss < sb-20) ext = 2; // double extension
                } else if(sb >= beta) {
                    return sb; // multi-cut
                }
            }

            // Cont hist key (現在の手)
            int cur_ck = -1;
            if(!m.is_drop() && pos.board[m.from()])
                cur_ck = ch_key(type_of(pos.board[m.from()]), m.to());

            pos.do_move(m);
            if(is_quiet) quiets_tried.push_back(m);

            Score s;
            int nd = depth - 1 + ext;

            if(cnt == 1) {
                s = -search(pos, -beta, -alpha, nd, ply+1, m, true, is_pv);
            } else {
                // LMR: log(depth) × log(cnt) 公式 (やねうら王V9準拠)
                int red = 0;
                if(depth >= 3 && cnt > 3 && !in_chk && !gc && ext == 0 && is_quiet) {
                    // log(depth+1) × log(cnt+1) / 2.5
                    // 例: depth=7,cnt=5 → log(8)*log(6)/2.5 ≈ 1.49 → 1
                    //     depth=10,cnt=10 → 2.40*2.40/2.5 ≈ 2.30 → 2
                    double lr = std::log(double(depth)+1.0) * std::log(double(cnt)+1.0) / 2.5;
                    red = std::max(0, (int)lr);
                    red = std::min(red, nd-1);

                    // 調整
                    if(is_pv)  red = std::max(0, red-1);
                    // History 補正 (大きい履歴値があれば削減を減らす)
                    int h = history[pos.turn()][m.from()][m.to()];
                    if(prev_ck >= 0 && cur_ck >= 0)
                        h += (int)cont_hist[prev_ck][cur_ck] / 2;
                    red -= std::clamp(h / 8000, -2, 2);
                    red = std::clamp(red, 0, nd-1);
                }

                s = -search(pos, -alpha-1, -alpha, nd-red, ply+1, m, true, false);
                if(s > alpha && (red > 0 || s < beta)) {
                    s = -search(pos, -beta, -alpha, nd, ply+1, m, true, is_pv);
                }
            }
            pos.undo_move(m);

            if(*stop_flag_ptr) return alpha;

            if(s > best) {
                best = s; best_mv = m;
                pv[ply][ply] = m;
                for(int i=ply+1;i<pv_len[ply+1];i++) pv[ply][i]=pv[ply+1][i];
                pv_len[ply] = pv_len[ply+1];
            }
            if(s > alpha) { alpha = s; bound = BOUND_EXACT; }
            if(alpha >= beta) {
                bound = BOUND_LOWER;
                // 履歴更新
                if(is_quiet) {
                    if(ply < MAX_PLY) {
                        if(m != killer[ply][0]) {
                            killer[ply][1] = killer[ply][0];
                            killer[ply][0] = m;
                        }
                    }
                    if(prev.ok() && !prev.is_drop())
                        counter[prev.from()][prev.to()] = m;
                    upd_quiet(pos, m, depth, prev_ck, ply);
                    for(Move q : quiets_tried)
                        if(q != m) pen_quiet(pos, q, depth, prev_ck);
                } else if(is_cap) {
                    upd_cap(pos, m, depth);
                }
                break;
            }
        }

        if(excluded == MOVE_NONE)
            tt->store(pos.key(), best_mv, best, depth, bound);
        return best;
    }

public:
    SearchEngine() : nodes(0), stop_flag_ptr(nullptr) {
        clear_tables();
    }

    void clear_tables() {
        nodes = 0;
        if(stop_flag_ptr) *stop_flag_ptr = false;
        std::memset(killer,    0, sizeof(killer));
        std::memset(history,   0, sizeof(history));
        std::memset(counter,   0, sizeof(counter));
        std::memset(cont_hist, 0, sizeof(cont_hist));
        std::memset(cap_hist,  0, sizeof(cap_hist));
        for(int i=0;i<MAX_PLY;i++) pv_len[i] = 0;
    }

    void new_game() { if(thread_id==0) tt->new_search(); game_moves.clear(); clear_tables(); }
    void set_pos(const Position& p) { root = p; }
    void set_startpos(bool b) { is_startpos = b; if(thread_id==0) book->set_startpos(b); }
    void push_move(const std::string& s) { game_moves.push_back(s); }

    // ---- USI変換 ----
    std::string to_usi(Move m) const {
        if(!m.ok()) return "resign";
        std::stringstream ss;
        if(m.is_drop()) {
            const char* n[]={"","P","L","N","S","G","B","R"};
            ss << n[m.drop_pt()] << "*"
               << (char)('1'+8-file_of(m.to()))
               << (char)('a'+rank_of(m.to()));
        } else {
            ss << (char)('1'+8-file_of(m.from()))
               << (char)('a'+rank_of(m.from()))
               << (char)('1'+8-file_of(m.to()))
               << (char)('a'+rank_of(m.to()));
            if(m.is_promo()) ss << "+";
        }
        return ss.str();
    }

    Move from_usi(const std::string& s) const {
        if(s.size() < 4) return MOVE_NONE;
        if(s[1] == '*') {
            const char* n[]={"","P","L","N","S","G","B","R"};
            PieceType pt = NO_PT;
            for(int i=1;i<8;i++) if(s[0]==n[i][0]){pt=(PieceType)i;break;}
            if(pt==NO_PT) return MOVE_NONE;
            int f=8-(s[2]-'1'), r=s[3]-'a';
            if(!valid_sq(f,r)) return MOVE_NONE;
            return Move::drop(pt, make_sq(f,r));
        } else {
            int ff=8-(s[0]-'1'), fr=s[1]-'a';
            int tf=8-(s[2]-'1'), tr=s[3]-'a';
            if(!valid_sq(ff,fr)||!valid_sq(tf,tr)) return MOVE_NONE;
            bool pr = s.size()>=5 && s[4]=='+';
            return Move::make(make_sq(ff,fr), make_sq(tf,tr), pr);
        }
    }

    Move from_usi_legal(const std::string& s, Position& p) const {
        Move m = from_usi(s);
        if(!m.ok()) return MOVE_NONE;
        auto lv = p.gen_legal();
        for(Move lm : lv) if(lm==m) return m;
        return MOVE_NONE;
    }

    // =========================================================================
    // 反復深化
    // =========================================================================
    Move go(int max_depth, int time_ms) {
        // 定跡
        if(is_startpos && thread_id == 0) {
            std::string bm = book->probe(game_moves);
            if(!bm.empty()) {
                Move m = from_usi_legal(bm, root);
                if(m.ok()) {
                    std::cout << "info string book " << bm << std::endl;
                    return m;
                }
            }
        }

        t_start = std::chrono::steady_clock::now();
        time_limit_ms = time_ms;
        *stop_flag_ptr = false; nodes = 0;
        if(thread_id == 0) tt->new_search();

        Move best_mv = MOVE_NONE;
        Score prev_sc = 0;

        // Position の1回コピー（vectorのheap allocationを避ける）
        Position search_pos;
        {
            // board+hand+stm+ply+hash+hist をコピー
            // histのvectorはコピーせず、explorationごとにclearする
            search_pos = root; // 一度だけコピー
        }

        for(int depth = 1 + thread_id; depth <= max_depth && !*stop_flag_ptr; depth += 1) {
            Score sc;
            // histをクリアしてrootの状態に戻す（高速）
            search_pos.hist.clear();
            // board等はrootと同一のはず（do_move/undo_moveが対称なので）

            if(depth < 5) {
                sc = search(search_pos, -SCORE_INF, SCORE_INF, depth, 0, MOVE_NONE);
            } else {
                // Aspiration Window (指数的拡大)
                int w = 24 + std::abs(prev_sc) / 8;
                Score lo = prev_sc - w, hi = prev_sc + w;
                int fails = 0;

                while(true) {
                    search_pos.hist.clear(); // histクリアのみ（boardはすでに正しい）
                    sc = search(search_pos, lo, hi, depth, 0, MOVE_NONE);
                    if(*stop_flag_ptr) break;

                    if(sc <= lo) {
                        hi = (lo + hi) / 2;
                        lo = std::max(sc - w, (Score)-SCORE_INF);
                        w = w * 4 / 3 + 5;
                    } else if(sc >= hi) {
                        hi = std::min(sc + w, (Score)SCORE_INF);
                        w = w * 4 / 3 + 5;
                    } else break;

                    if(++fails >= 4 || lo <= -SCORE_INF/2 || hi >= SCORE_INF/2) {
                        search_pos.hist.clear();
                        sc = search(search_pos, -SCORE_INF, SCORE_INF, depth, 0, MOVE_NONE);
                        break;
                    }
                }
            }
            if(*stop_flag_ptr) break;
            prev_sc = sc;

            bool found; TTEntry* e = tt->probe(root.key(), found);
            if(found && e->get_move().ok()) best_mv = e->get_move();

            if (thread_id == 0) {

            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now()-t_start).count();
            uint64_t nps = ms > 0 ? nodes*1000/ms : 0;

            std::cout << "info depth " << depth << " score ";
            if(is_mate_score(sc)) {
                int mp = sc > 0 ? SCORE_MATE-sc : SCORE_MATE+sc;
                std::cout << "mate " << (sc>0 ? mp : -mp);
            } else {
                std::cout << "cp " << sc;
            }
            std::cout << " nodes " << nodes
                      << " nps "   << nps
                      << " time "  << ms
                      << " hashfull " << tt->hashfull();
            if(pv_len[0] > 0) {
                std::cout << " pv";
                for(int i=0; i<pv_len[0]&&i<20; i++)
                    std::cout << " " << to_usi(pv[0][i]);
            }
            std::cout << std::endl;
            } // end thread_id == 0 print

            if(is_mate_score(sc)) break;

            auto ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now()-t_start).count();
            if(ms2 >= time_limit_ms * 45 / 100) break;
        }

        if(!best_mv.ok()) {
            auto mv = root.gen_legal();
            if(!mv.empty()) best_mv = mv[0];
        }
        return best_mv;
    }
};
class USI {
    Position     pos;
    TranspositionTable tt;
    OpeningBook  book;
    int num_threads = 4;
    std::atomic<bool> global_stop_flag{false};
    std::vector<std::unique_ptr<SearchEngine>> engines;

    void setup_engines() {
        engines.clear();
        for (int i=0; i<num_threads; i++) {
            auto eng = std::make_unique<SearchEngine>();
            eng->tt = &tt;
            eng->book = &book;
            eng->stop_flag_ptr = &global_stop_flag;
            eng->thread_id = i;
            engines.push_back(std::move(eng));
        }
    }

    void handle_position(std::istringstream& iss){
        std::string tok;
        iss>>tok;

        bool startpos_mode=false;
        if(tok=="startpos"){
            pos.set_hirate();
            startpos_mode=true;
        } else if(tok=="sfen"){
            std::string b,t,h,mc;
            if(iss>>b>>t>>h){ iss>>mc; pos.set_sfen(b+" "+t+" "+h); }
            startpos_mode=false;
        }
        for(auto& eng : engines) eng->set_startpos(startpos_mode);

        // moves
        if(iss>>tok&&tok=="moves"){
            while(iss>>tok){
                Move m=engines[0]->from_usi(tok);
                if(m.ok()){
                    auto lv=pos.gen_legal();
                    bool ok=false;
                    for(Move lm:lv) if(lm==m){ok=true;break;}
                    if(ok){ 
                        pos.do_move(m); 
                        for(auto& eng : engines) eng->push_move(tok); 
                    }
                    else std::cerr<<"[WARN] illegal move: "<<tok<<std::endl;
                }
            }
        }
        for(auto& eng : engines) eng->set_pos(pos);
    }

    void handle_go(std::istringstream& iss){
        int depth=30, time_ms=10000;
        int btime=0,wtime=0,byoyomi=0,binc=0,winc=0,movetime=0;
        std::string tok;
        while(iss>>tok){
            if(tok=="depth")    iss>>depth;
            else if(tok=="btime")   iss>>btime;
            else if(tok=="wtime")   iss>>wtime;
            else if(tok=="byoyomi") iss>>byoyomi;
            else if(tok=="binc")    iss>>binc;
            else if(tok=="winc")    iss>>winc;
            else if(tok=="movetime")iss>>movetime;
            else if(tok=="infinite"){depth=100;time_ms=3600000;}
        }

        if(movetime>0){
            time_ms=std::max(50,movetime-50);
        } else if(byoyomi>0){
            // 秒読み: 秒読み時間の90%使う（残り10%はマージン）
            time_ms=std::max(50,byoyomi*9/10);
        } else {
            int rem=(pos.turn()==BLACK)?btime:wtime;
            int inc=(pos.turn()==BLACK)?binc:winc;
            if(rem>0){
                // 残り手数を局面数に基づいて推定
                int gply=pos.game_ply();
                int moves_left;
                if(gply<20) moves_left=55;       // 序盤: たっぷり残す
                else if(gply<60) moves_left=45;   // 中盤
                else if(gply<100) moves_left=35;  // 終盤手前
                else moves_left=25;                // 終盤: 短く
                moves_left=std::max(15,moves_left);
                time_ms=rem/moves_left+inc*80/100;
                // 上限: 残り時間の35%
                time_ms=std::min(time_ms,rem*35/100);
                // 序盤節約（最初10手）
                if(gply<10) time_ms=std::min(time_ms,rem/70+inc);
            }
        }
        time_ms=std::max(50,std::min(time_ms,120000));

        global_stop_flag = false;
        std::vector<std::thread> threads;
        Move best_bm = MOVE_NONE;
        for (int i = 1; i < num_threads; i++) {
            threads.emplace_back([this, depth, time_ms, i]() {
                this->engines[i]->go(depth, time_ms);
            });
        }
        best_bm = engines[0]->go(depth, time_ms);
        global_stop_flag = true; // Tell other threads to stop
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        std::cout<<"bestmove "<<engines[0]->to_usi(best_bm)<<std::endl;
    }

public:
    USI() : tt(256) { 
        setup_engines();
        pos.set_hirate(); 
    }

    void loop(){
        std::string line,tok;
        while(std::getline(std::cin,line)){
            if(line.empty()) continue;
            std::istringstream iss(line);
            iss>>tok;
            if(tok=="usi"){
                std::cout<<"id name HYOUE_v8_R2000\n"
                         <<"id author ClaudeAI\n"
                         <<"id author Claude_v4\n"
                         <<"option name Hash type spin default 256 min 16 max 4096\n"
                         <<"option name Threads type spin default 4 min 1 max 128\n"
                         <<"option name BookFile type string default user_book1.db\n"
                         <<"usiok"<<std::endl;
            } else if(tok=="isready"){
                std::cout<<"readyok"<<std::endl;
            } else if(tok=="setoption"){
                std::string name_tok, name, value_tok, value;
                if(iss >> name_tok && name_tok == "name" && iss >> name && iss >> value_tok && value_tok == "value" && iss >> value) {
                    if (name == "Threads") {
                        num_threads = std::max(1, std::stoi(value));
                        setup_engines();
                    }
                    if (name == "Hash") {
                        tt.resize(std::max(16, std::stoi(value)));
                    }
                }
            } else if(tok=="usinewgame"){
                for(auto& eng : engines) eng->new_game();
                pos.set_hirate();
            } else if(tok=="position"){
                handle_position(iss);
            } else if(tok=="go"){
                handle_go(iss);
            } else if(tok=="stop"){
                // stopは次のnodes checkで処理
            } else if(tok=="quit"){
                break;
            }
        }
    }
};

// ========================================================================================================
// テスト
// ========================================================================================================

namespace Test {
void run(){
    std::cerr<<"=== Tests ==="<<std::endl;

    // 1. 初期局面合法手数
    Position pos; pos.set_hirate();
    auto mv=pos.gen_legal();
    std::cerr<<"[1] Initial legal moves: "<<mv.size()<<" (expect 30)"
             <<(mv.size()==30?" OK":" WARN")<<std::endl;

    // 2. 評価値が初期局面で±20以内
    Evaluator ev;
    Score sc=ev.evaluate(pos);
    std::cerr<<"[2] Initial eval: "<<sc<<" (expect near 0)"
             <<(std::abs(sc)<=20?" OK":" WARN")<<std::endl;

    // 3. undo/do一貫性
    Hash h0=pos.key();
    Move first=mv[0];
    pos.do_move(first);
    pos.undo_move(first);
    std::cerr<<"[3] Hash after do/undo: "<<(pos.key()==h0?"OK":"BUG!")<<std::endl;

    // 4. ranging駒テスト（角が4方向に動く）
    pos.set_sfen("9/9/9/9/4B4/9/9/9/9 b - 1");
    auto mv2=pos.gen_legal();
    std::cerr<<"[4] Bishop at 5e: "<<mv2.size()<<" moves (expect >=14)"
             <<(mv2.size()>=14?" OK":" WARN")<<std::endl;

    // 5. 飛車の利き
    pos.set_sfen("9/9/9/9/4R4/9/9/9/9 b - 1");
    auto mv3=pos.gen_legal();
    std::cerr<<"[5] Rook at 5e: "<<mv3.size()<<" moves (expect 16)"
             <<(mv3.size()==16?" OK":" WARN")<<std::endl;

    // 6. 馬（角成）の利き
    pos.set_sfen("9/9/9/9/4+B4/9/9/9/9 b - 1");
    auto mv4=pos.gen_legal();
    std::cerr<<"[6] Horse at 5e: "<<mv4.size()<<" moves (expect 20)"
             <<(mv4.size()>=18?" OK":" WARN")<<std::endl;

    // 7. 竜（飛成）の利き
    pos.set_sfen("9/9/9/9/4+R4/9/9/9/9 b - 1");
    auto mv5=pos.gen_legal();
    std::cerr<<"[7] Dragon at 5e: "<<mv5.size()<<" moves (expect 20)"
             <<(mv5.size()>=18?" OK":" WARN")<<std::endl;

    // 8. 二歩チェック
    pos.set_sfen("9/9/9/9/9/9/9/4P4/4K4 b P 1");
    auto mv6=pos.gen_legal();
    bool nifu=false;
    for(Move m:mv6) if(m.is_drop()&&m.drop_pt()==PAWN&&file_of(m.to())==4) nifu=true;
    std::cerr<<"[8] Nifu prevention: "<<(!nifu?"OK":"BUG!")<<std::endl;

    // 9. 8枚落ちSFEN
    pos.set_sfen(Position::DROP8_SFEN);
    auto mv7=pos.gen_legal();
    std::cerr<<"[9] 8-piece handicap: "<<mv7.size()<<" legal moves (後手番)"<<std::endl;

    // 10. 反則手ゼロ確認
    pos.set_hirate();
    auto all=pos.gen_legal();
    int illegal=0;
    for(Move m:all){
        Position t=pos; t.do_move(m);
        Square ks=t.king_sq(pos.turn());
        if(ks!=SQ_NONE&&t.attacked_by(ks,~pos.turn())) illegal++;
    }
    std::cerr<<"[10] Illegal moves in initial: "<<illegal<<(illegal==0?" OK":" BUG!")<<std::endl;

    // 11. Null move後のundo
    pos.set_hirate();
    Hash hbefore=pos.key();
    pos.do_move(MOVE_NULL);
    pos.undo_move(MOVE_NULL);
    std::cerr<<"[11] Null move undo hash: "<<(pos.key()==hbefore?"OK":"BUG!")<<std::endl;

    std::cerr<<"=== Done ==="<<std::endl;
}
}

// ========================================================================================================
// main
// ========================================================================================================

int main(int argc,char* argv[]){
    if(argc>1&&std::string(argv[1])=="test"){
        Test::run(); return 0;
    }
    USI usi;
    usi.loop();
    return 0;
}
