"""
查询执行器模块
执行SQL查询并格式化结果
"""
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


class QueryExecutor:
    """查询执行器"""
    
    def __init__(self, database_uri: str):
        """
        初始化查询执行器
        
        Args:
            database_uri: 数据库连接URI
        """
        self.engine: Engine = create_engine(database_uri)
    
    def execute_query(self, sql: str) -> Tuple[bool, Any, Optional[str]]:
        """
        执行SQL查询
        
        Args:
            sql: SQL查询语句
            
        Returns:
            (是否成功, 结果数据, 错误信息)
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql))
                
                # 转换为DataFrame以便后续处理
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                return True, df, None
                
        except SQLAlchemyError as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"执行错误: {str(e)}"
    
    def format_results(
        self, 
        df: pd.DataFrame, 
        format_type: str = "natural"
    ) -> str:
        """
        格式化查询结果
        
        Args:
            df: 查询结果DataFrame
            format_type: 格式类型 (natural/table/json)
            
        Returns:
            格式化后的结果字符串
        """
        if df.empty:
            return "查询未返回任何结果。"
        
        if format_type == "table":
            return self._format_as_table(df)
        elif format_type == "json":
            return df.to_json(orient="records", indent=2, force_ascii=False)
        else:  # natural
            return self._format_as_natural_language(df)
    
    def _format_as_table(self, df: pd.DataFrame) -> str:
        """格式化为表格"""
        return df.to_string(index=False)
    
    def _format_as_natural_language(self, df: pd.DataFrame) -> str:
        """格式化为自然语言描述"""
        rows, cols = df.shape
        
        result = f"查询返回了 {rows} 行数据"
        
        if rows == 0:
            return result + "。"
        
        result += f"，包含以下列: {', '.join(df.columns)}\n\n"
        
        # 显示前几行数据
        display_rows = min(rows, 10)
        result += f"前 {display_rows} 行数据:\n"
        result += df.head(display_rows).to_string(index=False)
        
        if rows > display_rows:
            result += f"\n\n... 还有 {rows - display_rows} 行数据"
        
        # 添加一些统计信息
        result += "\n\n数据摘要:"
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                result += f"\n- {col}: 平均值={df[col].mean():.2f}, "
                result += f"最小值={df[col].min()}, 最大值={df[col].max()}"
        
        return result
    
    def get_result_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取结果摘要
        
        Args:
            df: 查询结果DataFrame
            
        Returns:
            结果摘要字典
        """
        if df.empty:
            return {
                "row_count": 0,
                "column_count": 0,
                "columns": []
            }
        
        summary = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # 添加数值列的统计信息
        numeric_stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_stats[col] = {
                    "mean": float(df[col].mean()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "std": float(df[col].std())
                }
        
        if numeric_stats:
            summary["numeric_stats"] = numeric_stats
        
        return summary

