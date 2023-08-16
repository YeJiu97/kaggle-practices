# ===== libraries ========
library("ggplot2")
library(corrplot)
library(rpart)
library(randomForest)
library(naivebayes)
library(dplyr)
library(e1071)
library(caret)



# ====== 导入一些数据集 ======
titanic_train <- read.csv("train.csv")
titanic_test <- read.csv("test.csv")
titanic_test_true_result <- read.csv("gender_submission.csv")



# ===== 探索titanic_train数据集 =====
str(titanic_train)
summary(titanic_train)

str(titanic_train)
summary(titanic_test)



# ===== 将数据集中的变量类型修改了 ========
titanic_train$Survived <- factor(titanic_train$Survived, labels = c("No", "Yes"))
titanic_train$Pclass <- factor(titanic_train$Pclass, labels = c("1st", "2nd", "3rd"))
titanic_train$Sex <- factor(titanic_train$Sex)
titanic_train$Embarked <- factor(titanic_train$Embarked)

titanic_test$Pclass <- factor(titanic_test$Pclass, labels = c("1st", "2nd", "3rd"))
titanic_test$Sex <- factor(titanic_test$Sex)
titanic_test$Embarked <- factor(titanic_test$Embarked)

titanic_test_true_result$Survived <- factor(titanic_test_true_result$Survived, labels = c("No", "Yes"))



# ====== 检测数据的缺失值 =========
train_missing_values <- sum(is.na(titanic_train))
print(train_missing_values)
test_missing_values <- sum(is.na(titanic_test))
print(test_missing_values)



# ===== 填补缺失值 ================
titanic_train$Age[is.na(titanic_train$Age)]  <- mean(titanic_train$Age, na.rm = TRUE)
titanic_test$Age[is.na(titanic_test$Age)]  <- mean(titanic_test$Age, na.rm = TRUE)

titanic_train$Fare[is.na(titanic_train$Fare)]  <- mean(titanic_train$Age, na.rm = TRUE)
titanic_test$Fare[is.na(titanic_test$Fare)]  <- mean(titanic_test$Fare, na.rm = TRUE)



# ==== 再次查看是否还存在着缺失值
train_missing_values <- sum(is.na(titanic_train))
print(train_missing_values)
test_missing_values <- sum(is.na(titanic_test))
print(test_missing_values)



# ==== 对单变量进行探索 ==========
# age的直方图
ggplot(data = titanic_train, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "dodgerblue", color = "black") +
  labs(title = "Age Distribution",
       x = "Age",
       y = "Frequency")

hist(titanic_train$Age,
     main = "Age Distribution",
     xlab = "Age",
     ylab = "Frequency",
     col = "dodgerblue",
     border = "black")

boxplot(titanic_train$Age, 
        main = "Age Distribution",
        ylab = "Age",
        col = "dodgerblue",
        border = "black")

ggplot(data = titanic_train, aes(y = Age)) +
  geom_boxplot(fill = "dodgerblue", color = "black") +
  labs(title = "Age Distribution",
       y = "Age")

# survive的bar chart
ggplot(data = titanic_train, aes(x = Survived)) +
  geom_bar(binwidth = 5, fill = "dodgerblue", color = "black") +
  labs(title = "Survived Distribution",
       x = "Survived",
       y = "Frequency")

# Pclass的bar chart
ggplot(data = titanic_train, aes(x = Pclass)) +
  geom_bar(binwidth = 5, fill = "dodgerblue", color = "black") +
  labs(title = "Pclass Distribution",
       x = "Pclass",
       y = "Frequency")

# 同理还有sex的分布
ggplot(data = titanic_train, aes(x = Sex)) +
  geom_bar(binwidth = 5, fill = "dodgerblue", color = "black") +
  labs(title = "Sex Distribution",
       x = "Sex",
       y = "Frequency")

# 同理
ggplot(data = titanic_train, aes(x = SibSp)) +
  geom_bar(binwidth = 5, fill = "dodgerblue", color = "black") +
  labs(title = "SibSp Distribution",
       x = "SibSp",
       y = "Frequency")

# Fare的分布情况
ggplot(data = titanic_train, aes(x = Fare)) +
  geom_histogram(binwidth = 5, fill = "dodgerblue", color = "black") +
  labs(title = "Fare Distribution",
       x = "Fare",
       y = "Frequency")

ggplot(data = titanic_train, aes(x = Fare)) +
  geom_density(binwidth = 5, fill = "dodgerblue", color = "black") +
  labs(title = "Fare Distribution",
       x = "Fare",
       y = "Frequency")



# ======= 对多个变量进行可视化 ==========
# 假设你的数据框名为 titanic_train，且你关注 Age、Fare 和 Pclass 这些变量
pairs(~ Age + Fare + Pclass, data = titanic_train, col = "dodgerblue")

# 假设你的数据框名为 titanic_train，且你关注 Age 和 Pclass 这些变量
ggplot(data = titanic_train, aes(x = Pclass, y = Age)) +
  geom_violin(fill = "dodgerblue", color = "black") +
  labs(title = "Age Distribution by Pclass (Violin Plot)",
       x = "Pclass",
       y = "Age")

cor_matrix <- cor(titanic_train[c("Age", "Fare")])
corrplot(cor_matrix, method = "color")


# 计算相关性矩阵
cor_matrix <- cor(titanic_train[c("Age", "Fare", "SibSp", "Parch")])

# 绘制相关性矩阵热图
library(corrplot)
corrplot(cor_matrix, method = "color")



# ===== 开始训练模型 ================

# 逻辑回归模型
logic_model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = titanic_train,
             family = binomial)

# 接着是决策树模型
decision_tree_model <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
               data = titanic_train,
               method = "class")

# 接着是随机森林模型
random_forest_model <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                      data = titanic_train)

# 使用朴素贝叶斯模型
naive_bayes_model <- naive_bayes(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                     data = titanic_train,
                     laplace = 1)  # 设置拉普拉斯平滑参数

# 使用 svm 模型
# 将特征变量和目标变量分开
X_train <- as.matrix(select(titanic_train, -Survived))
y_train <- as.factor(titanic_train$Survived)

# 建立SVM模型
svm_model <- svm(y_train ~ ., data = data.frame(X_train), kernel = "linear")