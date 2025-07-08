# telegram_notifier.py - Telegram通知模块
# =============================================================================
# 核心职责：
# 1. 交易信号通知
# 2. 异常告警推送
# 3. 收益报告发送
# 4. 系统状态更新
# 5. 交互式命令处理
# =============================================================================

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from telegram import Bot, Update, ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, filters, ContextTypes
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from logger import TradingLogger
from utils import ConfigManager, TimeUtils, PriceUtils
from vnpy_integration import OrderResult, OrderStatus


@dataclass
class NotificationTemplate:
    """通知模板"""
    
    @staticmethod
    def trade_signal(signal: Dict[str, Any]) -> str:
        """交易信号模板"""
        emoji = "🟢" if signal['action'] == 'buy' else "🔴"
        return f"""
{emoji} *交易信号*

品种: {signal['symbol']}
方向: {signal['action'].upper()}
数量: {signal['size']}
置信度: {signal['confidence']:.2%}
模型: {signal['model_type']}
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    @staticmethod
    def order_result(result: OrderResult) -> str:
        """订单结果模板"""
        status_emoji = {
            OrderStatus.FILLED: "✅",
            OrderStatus.PARTIAL_FILLED: "⚠️",
            OrderStatus.CANCELLED: "❌",
            OrderStatus.REJECTED: "🚫",
            OrderStatus.FAILED: "❌"
        }
        
        emoji = status_emoji.get(result.status, "❓")
        
        text = f"""
{emoji} *订单更新*

品种: {result.signal.symbol}
方向: {result.signal.action.upper()}
状态: {result.status.value}
"""
        
        if result.filled_size > 0:
            text += f"""
成交数量: {result.filled_size}
成交均价: {result.avg_price}
手续费: {result.commission}
滑点: {result.slippage:.4f}
"""
        
        if result.error_msg:
            text += f"\n错误信息: {result.error_msg}"
        
        return text
    
    @staticmethod
    def daily_report(stats: Dict[str, Any]) -> str:
        """日报模板"""
        return f"""
📊 *每日交易报告*

日期: {datetime.now().strftime('%Y-%m-%d')}

📈 *交易统计*
信号数量: {stats.get('signals_count', 0)}
执行订单: {stats.get('orders_count', 0)}
成功率: {stats.get('success_rate', 0):.1%}

💰 *收益情况*
今日盈亏: {stats.get('daily_pnl', 0):+.2f}
本月盈亏: {stats.get('monthly_pnl', 0):+.2f}
总盈亏: {stats.get('total_pnl', 0):+.2f}

📊 *持仓统计*
持仓品种: {stats.get('position_count', 0)}
总持仓价值: {stats.get('position_value', 0):.2f}

⚙️ *系统状态*
运行时长: {stats.get('uptime', 'N/A')}
CPU使用率: {stats.get('cpu_usage', 0):.1f}%
内存使用率: {stats.get('memory_usage', 0):.1f}%
"""
    
    @staticmethod
    def alert(alert_data: Dict[str, Any]) -> str:
        """告警模板"""
        level_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        emoji = level_emoji.get(alert_data['level'], '❓')
        
        return f"""
{emoji} *系统告警*

级别: {alert_data['level'].upper()}
组件: {alert_data['component']}
消息: {alert_data['message']}
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""


class TelegramNotifier:
    """
    Telegram通知器
    
    核心功能：
    1. 实时推送交易通知
    2. 定时发送报告
    3. 处理用户命令
    4. 管理订阅者
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = TradingLogger().get_logger("TELEGRAM_NOTIFIER")
        
        # Telegram配置
        self.token = self.config.get("notifications.telegram_token", "")
        self.chat_ids = self.config.get("notifications.telegram_chat_ids", [])
        self.admin_chat_id = self.config.get("notifications.telegram_admin_id", "")
        
        # Bot实例
        self.bot: Optional[Bot] = None
        self.application: Optional[Application] = None
        
        # 通知设置
        self.enable_trade_notifications = self.config.get(
            "notifications.enable_trade", True
        )
        self.enable_alert_notifications = self.config.get(
            "notifications.enable_alert", True
        )
        self.enable_daily_report = self.config.get(
            "notifications.enable_daily_report", True
        )
        self.report_time = self.config.get(
            "notifications.report_time", "20:00"
        )
        
        # 命令回调
        self.command_callbacks: Dict[str, Callable] = {}
        
        # 报告任务
        self.report_task = None
        
        self.logger.info("Telegram通知器初始化")
    
    async def initialize(self) -> bool:
        """初始化"""
        try:
            if not self.token:
                self.logger.warning("未配置Telegram token")
                return False
            
            # 创建Bot
            self.application = Application.builder().token(self.token).build()
            self.bot = self.application.bot
            
            # 注册命令处理器
            self._register_handlers()
            
            # 启动Bot
            await self.application.initialize()
            await self.application.start()
            
            # 发送启动通知
            await self._send_to_admin("🚀 STARK4系统已启动")
            
            # 启动定时报告
            if self.enable_daily_report:
                self.report_task = asyncio.create_task(self._daily_report_loop())
            
            self.logger.info("Telegram通知器初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            return False
    
    def _register_handlers(self):
        """注册命令处理器"""
        # 基础命令
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("report", self._cmd_report))
        
        # 交易命令
        self.application.add_handler(CommandHandler("positions", self._cmd_positions))
        self.application.add_handler(CommandHandler("pause", self._cmd_pause))
        self.application.add_handler(CommandHandler("resume", self._cmd_resume))
        
        # 配置命令
        self.application.add_handler(CommandHandler("subscribe", self._cmd_subscribe))
        self.application.add_handler(CommandHandler("unsubscribe", self._cmd_unsubscribe))
        
        # 回调查询处理
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))
    
    async def send_trade_notification(self, result: OrderResult):
        """发送交易通知"""
        if not self.enable_trade_notifications:
            return
        
        try:
            message = NotificationTemplate.order_result(result)
            
            # 添加操作按钮
            keyboard = [[
                InlineKeyboardButton("查看详情", callback_data=f"order_{result.order_id}"),
                InlineKeyboardButton("查看持仓", callback_data="positions")
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self._broadcast(
                message, 
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f"发送交易通知失败: {e}")
    
    async def send_signal_notification(self, signal: Dict[str, Any]):
        """发送信号通知"""
        if not self.enable_trade_notifications:
            return
        
        try:
            message = NotificationTemplate.trade_signal(signal)
            await self._broadcast(message, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"发送信号通知失败: {e}")
    
    async def send_alert(self, alert_data: Dict[str, Any]):
        """发送告警"""
        if not self.enable_alert_notifications:
            return
        
        try:
            message = NotificationTemplate.alert(alert_data)
            
            # 严重告警发送给管理员
            if alert_data.get('level') in ['error', 'critical']:
                await self._send_to_admin(message, parse_mode=ParseMode.MARKDOWN)
            else:
                await self._broadcast(message, parse_mode=ParseMode.MARKDOWN)
                
        except Exception as e:
            self.logger.error(f"发送告警失败: {e}")
    
    async def send_daily_report(self, stats: Dict[str, Any]):
        """发送日报"""
        try:
            message = NotificationTemplate.daily_report(stats)
            
            # 添加操作按钮
            keyboard = [[
                InlineKeyboardButton("详细报告", callback_data="detailed_report"),
                InlineKeyboardButton("下载报告", callback_data="download_report")
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self._broadcast(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f"发送日报失败: {e}")
    
    async def _daily_report_loop(self):
        """定时报告循环"""
        while True:
            try:
                # 计算下次报告时间
                now = datetime.now()
                report_hour, report_minute = map(int, self.report_time.split(':'))
                
                next_report = now.replace(
                    hour=report_hour, 
                    minute=report_minute, 
                    second=0, 
                    microsecond=0
                )
                
                if next_report <= now:
                    next_report += timedelta(days=1)
                
                # 等待到报告时间
                wait_seconds = (next_report - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                # 生成并发送报告
                if 'generate_daily_report' in self.command_callbacks:
                    stats = await self.command_callbacks['generate_daily_report']()
                    await self.send_daily_report(stats)
                
            except Exception as e:
                self.logger.error(f"定时报告异常: {e}")
                await asyncio.sleep(3600)  # 出错后等待1小时
    
    async def _broadcast(self, message: str, **kwargs):
        """广播消息"""
        for chat_id in self.chat_ids:
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    **kwargs
                )
            except Exception as e:
                self.logger.error(f"发送到 {chat_id} 失败: {e}")
    
    async def _send_to_admin(self, message: str, **kwargs):
        """发送给管理员"""
        if self.admin_chat_id:
            try:
                await self.bot.send_message(
                    chat_id=self.admin_chat_id,
                    text=message,
                    **kwargs
                )
            except Exception as e:
                self.logger.error(f"发送给管理员失败: {e}")
    
    # 命令处理器
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/start命令"""
        await update.message.reply_text(
            "欢迎使用STARK4智能量化交易系统！\n"
            "使用 /help 查看可用命令。"
        )
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/help命令"""
        help_text = """
*可用命令:*

/status - 查看系统状态
/report - 获取交易报告
/positions - 查看当前持仓
/pause - 暂停交易
/resume - 恢复交易
/subscribe - 订阅通知
/unsubscribe - 取消订阅

*管理员命令:*
/restart - 重启系统
/shutdown - 关闭系统
/config - 查看配置
        """
        await update.message.reply_text(
            help_text, 
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/status命令"""
        if 'get_system_status' in self.command_callbacks:
            status = await self.command_callbacks['get_system_status']()
            
            status_text = f"""
*系统状态*

运行状态: {status.get('state', 'Unknown')}
运行时长: {status.get('uptime', 'N/A')}

*组件状态:*
"""
            for component, is_running in status.get('components', {}).items():
                emoji = "✅" if is_running else "❌"
                status_text += f"{emoji} {component}\n"
            
            await update.message.reply_text(
                status_text,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text("无法获取系统状态")
    
    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/report命令"""
        if 'generate_report' in self.command_callbacks:
            report = await self.command_callbacks['generate_report']()
            await update.message.reply_text(
                report,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text("无法生成报告")
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/positions命令"""
        if 'get_positions' in self.command_callbacks:
            positions = await self.command_callbacks['get_positions']()
            
            if not positions:
                await update.message.reply_text("当前无持仓")
                return
            
            text = "*当前持仓:*\n\n"
            for symbol, pos_data in positions.items():
                text += f"{symbol}: {pos_data['volume']} @ {pos_data['price']}\n"
                text += f"盈亏: {pos_data['pnl']:+.2f}\n\n"
            
            await update.message.reply_text(
                text,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text("无法获取持仓信息")
    
    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/pause命令"""
        # 检查权限
        if str(update.effective_user.id) != self.admin_chat_id:
            await update.message.reply_text("您没有权限执行此操作")
            return
        
        if 'pause_trading' in self.command_callbacks:
            await self.command_callbacks['pause_trading']()
            await update.message.reply_text("交易已暂停")
        else:
            await update.message.reply_text("无法执行操作")
    
    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/resume命令"""
        # 检查权限
        if str(update.effective_user.id) != self.admin_chat_id:
            await update.message.reply_text("您没有权限执行此操作")
            return
        
        if 'resume_trading' in self.command_callbacks:
            await self.command_callbacks['resume_trading']()
            await update.message.reply_text("交易已恢复")
        else:
            await update.message.reply_text("无法执行操作")
    
    async def _cmd_subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/subscribe命令"""
        chat_id = str(update.effective_chat.id)
        
        if chat_id not in self.chat_ids:
            self.chat_ids.append(chat_id)
            # 保存配置
            self.config.set("notifications.telegram_chat_ids", self.chat_ids)
            self.config.save_config()
            
            await update.message.reply_text("已订阅通知")
        else:
            await update.message.reply_text("您已经订阅了通知")
    
    async def _cmd_unsubscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理/unsubscribe命令"""
        chat_id = str(update.effective_chat.id)
        
        if chat_id in self.chat_ids:
            self.chat_ids.remove(chat_id)
            # 保存配置
            self.config.set("notifications.telegram_chat_ids", self.chat_ids)
            self.config.save_config()
            
            await update.message.reply_text("已取消订阅")
        else:
            await update.message.reply_text("您未订阅通知")
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理回调查询"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "positions":
            await self._cmd_positions(update, context)
        elif data == "detailed_report":
            # 实现详细报告逻辑
            await query.message.reply_text("详细报告功能开发中...")
        elif data.startswith("order_"):
            # 实现订单详情逻辑
            order_id = data.replace("order_", "")
            await query.message.reply_text(f"订单 {order_id} 详情开发中...")
    
    def register_callback(self, name: str, callback: Callable):
        """注册回调函数"""
        self.command_callbacks[name] = callback
    
    async def shutdown(self):
        """关闭通知器"""
        try:
            # 发送关闭通知
            await self._send_to_admin("⚠️ STARK4系统正在关闭")
            
            # 取消定时任务
            if self.report_task:
                self.report_task.cancel()
            
            # 关闭Bot
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
            
            self.logger.info("Telegram通知器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭失败: {e}")
