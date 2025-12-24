import matplotlib
# 设置Agg后端（无GUI环境）
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

# ==================== 字体设置 ====================
def setup_chinese_font():
    """设置中文字体，解决警告问题"""
    try:
        # 尝试使用系统字体
        system_fonts = fm.findSystemFonts()
        
        # 常见中文字体名称
        chinese_font_candidates = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK',
            'Droid Sans Fallback',
            'DejaVu Sans',
            'Arial Unicode MS',
            'Microsoft YaHei',
            'SimHei',
            'SimSun'
        ]
        
        # 查找可用的中文字体
        available_fonts = []
        for font_path in system_fonts:
            try:
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                if any(candidate in font_name for candidate in chinese_font_candidates):
                    available_fonts.append((font_name, font_path))
            except:
                continue
        
        if available_fonts:
            # 使用第一个找到的中文字体
            font_name, font_path = available_fonts[0]
            # 添加到matplotlib
            fm.fontManager.addfont(font_path)
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"✅ 使用字体: {font_name}")
        else:
            # 如果找不到中文字体，使用默认字体
            print("⚠️  未找到中文字体，将使用默认字体")
            # 设置字体为默认英文字体，避免中文警告
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
            matplotlib.rcParams['axes.unicode_minus'] = False
            
    except Exception as e:
        print(f"⚠️  字体设置失败: {e}")
        # 设置回退方案
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        matplotlib.rcParams['axes.unicode_minus'] = False

# 初始化字体设置
setup_chinese_font()

# ==================== 数据处理函数 ====================
def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 15 Dec 2025\n\n现在你要扮演神话角色——孙悟空<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ==================== 自定义Trainer ====================
class CustomTrainer(Trainer):
    """自定义Trainer以记录训练过程中的损失"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        self.best_eval_loss = float('inf')
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # 计算损失
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        # 记录训练损失（只在训练模式下）
        if model.training:
            self.train_losses.append(loss.item())
            self.train_steps.append(self.state.global_step)
            
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(self, *args, **kwargs):
        # 执行评估并记录评估损失
        output = super().evaluation_loop(*args, **kwargs)
        
        if output.metrics.get("eval_loss") is not None:
            eval_loss = output.metrics["eval_loss"]
            self.eval_losses.append(eval_loss)
            self.eval_steps.append(self.state.global_step)
            
            # 保存最佳模型
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                # 保存最佳模型
                best_model_path = os.path.join(self.args.output_dir, "best_model")
                self.save_model(best_model_path)
                print(f"\n✨ 新的最佳模型已保存，验证损失: {eval_loss:.4f}")
                
        return output
    
    # def plot_loss_curves(self, output_dir):
    #     """绘制损失曲线并保存"""
    #     plt.figure(figsize=(12, 6))
        
    #     # 绘制训练损失
    #     if self.train_losses:
    #         # 平滑训练损失曲线
    #         smooth_loss = np.convolve(self.train_losses, np.ones(10)/10, mode='valid')
    #         smooth_steps = self.train_steps[:len(smooth_loss)]
    #         plt.plot(smooth_steps, smooth_loss, 'b-', label='训练损失', alpha=0.7, linewidth=1.5)
            
    #     # 绘制验证损失
    #     if self.eval_losses and self.eval_steps:
    #         plt.plot(self.eval_steps, self.eval_losses, 'r-', label='验证损失', alpha=0.7, linewidth=2, marker='o')
            
    #     plt.xlabel('训练步数 (Steps)', fontsize=12)
    #     plt.ylabel('损失 (Loss)', fontsize=12)
    #     plt.title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    #     plt.legend(fontsize=11)
    #     plt.grid(True, alpha=0.3)
        
    #     # 保存损失曲线图
    #     loss_plot_path = os.path.join(output_dir, "loss_curves.png")
    #     plt.tight_layout()
    #     plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     print(f"📈 损失曲线已保存至: {loss_plot_path}")
        
    #     # 保存损失数据为JSON文件
    #     loss_data = {
    #         "train_losses": self.train_losses,
    #         "train_steps": self.train_steps,
    #         "eval_losses": self.eval_losses,
    #         "eval_steps": self.eval_steps,
    #         "best_eval_loss": self.best_eval_loss
    #     }
        
    #     loss_data_path = os.path.join(output_dir, "loss_data.json")
    #     with open(loss_data_path, 'w', encoding='utf-8') as f:
    #         json.dump(loss_data, f, ensure_ascii=False, indent=2)
        
    #     print(f"📊 损失数据已保存至: {loss_data_path}")
        
    #     return loss_data

    def plot_loss_curves(self, output_dir):
        """绘制损失曲线并保存"""
        plt.figure(figsize=(12, 6))
        
        # 使用英文标签
        if self.train_losses:
            plt.plot(self.train_steps, self.train_losses, 'b-', label='Training Loss', alpha=0.7, linewidth=2)
            
        if self.eval_losses and self.eval_steps:
            plt.plot(self.eval_steps, self.eval_losses, 'r-', label='Validation Loss', alpha=0.7, linewidth=2, marker='o')
            
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 保存损失曲线图
        loss_plot_path = os.path.join(output_dir, "loss_curves.png")
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Loss curves saved to: {loss_plot_path}")
        
        # 保存损失数据为JSON文件
        loss_data = {
            "train_losses": self.train_losses,
            "train_steps": self.train_steps,
            "eval_losses": self.eval_losses,
            "eval_steps": self.eval_steps,
            "best_eval_loss": self.best_eval_loss
        }
        
        loss_data_path = os.path.join(output_dir, "loss_data.json")
        with open(loss_data_path, 'w', encoding='utf-8') as f:
            json.dump(loss_data, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Loss data saved to: {loss_data_path}")
        
        return loss_data

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 1. 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        './LLM-Research/Meta-Llama-3___1-8B', 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(
        './LLM-Research/Meta-Llama-3___1-8B', 
        use_fast=False, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载并处理数据
    df = pd.read_json('chat_wukong1.json')
    
    # 3. 划分训练集和验证集 (80%训练, 20%验证)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"📊 数据集统计:")
    print(f"  训练集样本数: {len(train_df)}")
    print(f"  验证集样本数: {len(eval_df)}")
    print(f"  总样本数: {len(df)}")
    
    # 创建训练集和验证集Dataset
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)
    
    # 处理数据
    tokenized_train = train_ds.map(process_func, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(process_func, remove_columns=eval_ds.column_names)
    
    # 4. 配置LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # 5. 配置训练参数
    output_dir = "./output/llama3_1_lora"
    
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        num_train_epochs=3,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        logging_dir=f"{output_dir}/logs",
        save_total_limit=3,
        dataloader_num_workers=4,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
    )
    
    # 6. 创建数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True,
        pad_to_multiple_of=8
    )
    
    # 7. 创建自定义Trainer
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # 8. 开始训练
    print("🚀 开始训练...")
    train_result = trainer.train()
    
    # 9. 保存最终模型
    trainer.save_model()
    trainer.save_state()
    
    # 10. 绘制损失曲线
    loss_data = trainer.plot_loss_curves(output_dir)
    
    # 11. 打印训练摘要
    print("\n" + "="*50)
    print("🏁 训练完成!")
    print("="*50)
    print(f"最佳验证损失: {trainer.best_eval_loss:.4f}")
    print(f"最终训练损失: {train_result.training_loss:.4f}")
    print(f"模型保存在: {output_dir}")
    print(f"最佳模型保存在: {os.path.join(output_dir, 'best_model')}")
    print("="*50)
    
    # 12. 可选：在验证集上进行最终评估
    print("\n📋 在验证集上进行最终评估...")
    eval_results = trainer.evaluate()
    print(f"最终验证损失: {eval_results['eval_loss']:.4f}")