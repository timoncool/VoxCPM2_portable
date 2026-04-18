@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   VoxCPM2 Portable - Обновление
echo ========================================
echo.

REM Определяем директорию скрипта
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Проверяем наличие git
where git >nul 2>nul
if errorlevel 1 (
    echo ОШИБКА: Git не найден!
    echo.
    echo Для обновления необходим Git.
    echo Скачайте его с https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

REM Проверяем что это git репозиторий
if not exist ".git" (
    echo ОШИБКА: Это не git репозиторий!
    echo.
    echo Обновление возможно только для версии, склонированной через git.
    echo.
    pause
    exit /b 1
)

echo Получение обновлений с GitHub...
echo.

git pull

if errorlevel 1 (
    echo.
    echo ========================================
    echo ОШИБКА при обновлении!
    echo ========================================
    echo.
    echo Возможные причины:
    echo 1. Нет подключения к интернету
    echo 2. Локальные изменения конфликтуют с обновлениями
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Обновление завершено успешно!
echo ========================================
echo.
echo Для запуска приложения используйте: run.bat
echo.
pause
