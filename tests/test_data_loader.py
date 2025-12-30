"""
tests/test_data_loader.py - data_loader 模块单元测试

测试数据加载、清洗和字段标准化功能。
"""

import json
import tempfile
import pytest
from pathlib import Path

import pandas as pd

# 添加 src 到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import (
    load_chat_data,
    get_session_info,
    _filter_message_types,
    _clean_content,
    _remove_xml_tags,
    _standardize_fields,
    _add_time_columns
)


# ==================== 测试数据 ====================

SAMPLE_CHAT_DATA = {
    "session": {
        "displayName": "测试群聊",
        "wxid": "test_group@chatroom",
        "messageCount": 5
    },
    "messages": [
        {
            "type": "文本消息",
            "content": "大家好！",
            "createTime": 1704067200,  # 2024-01-01 00:00:00 UTC
            "senderDisplayName": "用户A",
            "senderUsername": "user_a"
        },
        {
            "type": "文本消息",
            "content": "你好呀~",
            "createTime": 1704067260,  # 2024-01-01 00:01:00 UTC
            "senderDisplayName": "用户B",
            "senderUsername": "user_b"
        },
        {
            "type": "引用消息",
            "content": "哈哈哈",
            "createTime": 1704067320,
            "senderDisplayName": "用户A",
            "senderUsername": "user_a"
        },
        {
            "type": "图片消息",  # 应该被过滤
            "content": "[图片]",
            "createTime": 1704067380,
            "senderDisplayName": "用户C",
            "senderUsername": "user_c"
        },
        {
            "type": "文本消息",
            "content": "<msg>XML内容</msg>真正的消息",
            "createTime": 1704067440,
            "senderDisplayName": "",  # 空昵称，应该使用 username
            "senderUsername": "user_d"
        }
    ]
}


@pytest.fixture
def sample_json_file():
    """创建临时测试 JSON 文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(SAMPLE_CHAT_DATA, f, ensure_ascii=False)
        return f.name


@pytest.fixture
def sample_df():
    """创建测试 DataFrame"""
    return pd.DataFrame(SAMPLE_CHAT_DATA['messages'])


# ==================== 测试用例 ====================

class TestLoadChatData:
    """测试 load_chat_data 函数"""
    
    def test_load_basic(self, sample_json_file):
        """测试基本加载功能"""
        df, session = load_chat_data(sample_json_file)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(session, dict)
        assert session['displayName'] == '测试群聊'
    
    def test_filters_non_text_messages(self, sample_json_file):
        """测试过滤非文本消息"""
        df, _ = load_chat_data(sample_json_file)
        
        # 原始 5 条消息，图片消息应被过滤
        assert len(df) == 4
    
    def test_has_required_columns(self, sample_json_file):
        """测试包含必需的列"""
        df, _ = load_chat_data(sample_json_file)
        
        required_columns = ['user', 'content', 'timestamp', 'hour', 'date', 'weekday']
        for col in required_columns:
            assert col in df.columns, f"缺少列: {col}"
    
    def test_empty_file_raises_error(self):
        """测试空消息文件抛出异常"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({"session": {}, "messages": []}, f)
            f.flush()
            
            with pytest.raises(ValueError, match="没有找到消息数据"):
                load_chat_data(f.name)


class TestRemoveXmlTags:
    """测试 _remove_xml_tags 函数"""
    
    def test_removes_simple_tag(self):
        """测试移除简单标签"""
        assert _remove_xml_tags("<msg>内容</msg>真正的消息") == "内容真正的消息"
    
    def test_handles_empty_string(self):
        """测试空字符串"""
        assert _remove_xml_tags("") == ""
    
    def test_handles_non_string(self):
        """测试非字符串输入"""
        assert _remove_xml_tags(None) == ""
        assert _remove_xml_tags(123) == ""
    
    def test_no_tags(self):
        """测试无标签文本"""
        assert _remove_xml_tags("普通消息") == "普通消息"


class TestFilterMessageTypes:
    """测试 _filter_message_types 函数"""
    
    def test_keeps_text_messages(self, sample_df):
        """测试保留文本消息"""
        result = _filter_message_types(sample_df)
        
        assert len(result) == 4  # 3 文本 + 1 引用
        assert all(result['type'].isin(['文本消息', '引用消息']))
    
    def test_filters_image_messages(self, sample_df):
        """测试过滤图片消息"""
        result = _filter_message_types(sample_df)
        
        assert '图片消息' not in result['type'].values


class TestStandardizeFields:
    """测试 _standardize_fields 函数"""
    
    def test_uses_display_name(self, sample_df):
        """测试使用显示名称"""
        result = _standardize_fields(sample_df)
        
        assert '用户A' in result['user'].values
    
    def test_fallback_to_username(self, sample_df):
        """测试昵称为空时使用用户名"""
        result = _standardize_fields(sample_df)
        
        # 最后一条消息的 senderDisplayName 为空
        assert 'user_d' in result['user'].values


class TestGetSessionInfo:
    """测试 get_session_info 函数"""
    
    def test_returns_session_dict(self, sample_json_file):
        """测试返回会话信息"""
        session = get_session_info(sample_json_file)
        
        assert session['displayName'] == '测试群聊'
        assert session['wxid'] == 'test_group@chatroom'


# ==================== 运行测试 ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
