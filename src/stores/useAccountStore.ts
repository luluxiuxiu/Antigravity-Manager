import { create } from 'zustand';
import { Account } from '../types/account';
import * as accountService from '../services/accountService';

// 定时刷新配额的间隔（毫秒）- 5 分钟
const QUOTA_REFRESH_INTERVAL = 5 * 60 * 1000;

// 全局定时器 ID
let quotaRefreshTimer: ReturnType<typeof setInterval> | null = null;

interface AccountState {
    accounts: Account[];
    currentAccount: Account | null;
    loading: boolean;
    error: string | null;
    isAutoRefreshing: boolean; // 是否正在自动刷新
    lastAutoRefreshTime: number | null; // 上次自动刷新时间

    // Actions
    fetchAccounts: () => Promise<void>;
    fetchCurrentAccount: () => Promise<void>;
    addAccount: (email: string, refreshToken: string) => Promise<void>;
    deleteAccount: (accountId: string) => Promise<void>;
    switchAccount: (accountId: string) => Promise<void>;
    refreshQuota: (accountId: string) => Promise<void>;
    refreshAllQuotas: () => Promise<accountService.RefreshStats>;

    // 定时刷新 actions
    startAutoRefresh: () => void;
    stopAutoRefresh: () => void;

    // 新增 actions
    startOAuthLogin: () => Promise<void>;
    cancelOAuthLogin: () => Promise<void>;
    importV1Accounts: () => Promise<void>;
    importFromDb: () => Promise<void>;
}

export const useAccountStore = create<AccountState>((set, get) => ({
    accounts: [],
    currentAccount: null,
    loading: false,
    error: null,
    isAutoRefreshing: false,
    lastAutoRefreshTime: null,

    fetchAccounts: async () => {
        set({ loading: true, error: null });
        try {
            console.log('[Store] Fetching accounts...');
            const accounts = await accountService.listAccounts();
            set({ accounts, loading: false });
        } catch (error) {
            console.error('[Store] Fetch accounts failed:', error);
            set({ error: String(error), loading: false });
        }
    },

    fetchCurrentAccount: async () => {
        set({ loading: true, error: null });
        try {
            const account = await accountService.getCurrentAccount();
            set({ currentAccount: account, loading: false });
        } catch (error) {
            set({ error: String(error), loading: false });
        }
    },

    addAccount: async (email: string, refreshToken: string) => {
        set({ loading: true, error: null });
        try {
            await accountService.addAccount(email, refreshToken);
            await get().fetchAccounts();
            set({ loading: false });
        } catch (error) {
            set({ error: String(error), loading: false });
            throw error;
        }
    },

    deleteAccount: async (accountId: string) => {
        set({ loading: true, error: null });
        try {
            await accountService.deleteAccount(accountId);
            await get().fetchAccounts();
            set({ loading: false });
        } catch (error) {
            set({ error: String(error), loading: false });
            throw error;
        }
    },

    switchAccount: async (accountId: string) => {
        set({ loading: true, error: null });
        try {
            await accountService.switchAccount(accountId);
            await get().fetchCurrentAccount();
            set({ loading: false });
        } catch (error) {
            set({ error: String(error), loading: false });
            throw error;
        }
    },

    refreshQuota: async (accountId: string) => {
        set({ loading: true, error: null });
        try {
            await accountService.fetchAccountQuota(accountId);
            await get().fetchAccounts();
            set({ loading: false });
        } catch (error) {
            set({ error: String(error), loading: false });
            throw error;
        }
    },

    refreshAllQuotas: async () => {
        set({ loading: true, error: null });
        try {
            const stats = await accountService.refreshAllQuotas();
            await get().fetchAccounts();
            set({ loading: false });
            return stats;
        } catch (error) {
            set({ error: String(error), loading: false });
            throw error;
        }
    },

    startOAuthLogin: async () => {
        set({ loading: true, error: null });
        try {
            await accountService.startOAuthLogin();
            await get().fetchAccounts();
            set({ loading: false });
        } catch (error) {
            set({ error: String(error), loading: false });
            throw error;
        }
    },

    cancelOAuthLogin: async () => {
        try {
            await accountService.cancelOAuthLogin();
            set({ loading: false, error: null });
        } catch (error) {
            console.error('[Store] Cancel OAuth failed:', error);
        }
    },

    importV1Accounts: async () => {
        set({ loading: true, error: null });
        try {
            await accountService.importV1Accounts();
            await get().fetchAccounts();
            set({ loading: false });
        } catch (error) {
            set({ error: String(error), loading: false });
            throw error;
        }
    },

    importFromDb: async () => {
        set({ loading: true, error: null });
        try {
            await accountService.importFromDb();
            await get().fetchAccounts();
            set({ loading: false });
        } catch (error) {
            set({ error: String(error), loading: false });
            throw error;
        }
    },

    // 启动定时自动刷新配额
    startAutoRefresh: () => {
        // 如果已经有定时器在运行，先停止
        if (quotaRefreshTimer) {
            console.log('[Store] 定时刷新已在运行，跳过');
            return;
        }

        console.log('[Store] 启动配额定时刷新 (间隔: 5分钟)');

        // 执行自动刷新的函数
        const doAutoRefresh = async () => {
            const state = get();
            // 如果正在手动刷新，跳过本次自动刷新
            if (state.loading || state.isAutoRefreshing) {
                console.log('[Store] 跳过自动刷新 (正在进行其他操作)');
                return;
            }

            console.log('[Store] 开始自动刷新配额...');
            set({ isAutoRefreshing: true });

            try {
                const stats = await accountService.refreshAllQuotas();
                await get().fetchAccounts();
                set({ 
                    isAutoRefreshing: false,
                    lastAutoRefreshTime: Date.now()
                });
                console.log(`[Store] 自动刷新完成: ${stats.success} 成功, ${stats.failed} 失败`);
            } catch (error) {
                console.error('[Store] 自动刷新失败:', error);
                set({ isAutoRefreshing: false });
            }
        };

        // 立即执行一次
        doAutoRefresh();

        // 设置定时器
        quotaRefreshTimer = setInterval(doAutoRefresh, QUOTA_REFRESH_INTERVAL);
    },

    // 停止定时自动刷新
    stopAutoRefresh: () => {
        if (quotaRefreshTimer) {
            console.log('[Store] 停止配额定时刷新');
            clearInterval(quotaRefreshTimer);
            quotaRefreshTimer = null;
        }
        set({ isAutoRefreshing: false });
    },
}));
