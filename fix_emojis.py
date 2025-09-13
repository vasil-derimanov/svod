import re

with open('video_orientation_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace common emojis with plain text
replacements = {
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    '📦': '[PACKAGE]',
    '📝': '[NOTE]',
    '⬇️': '[DOWNLOAD]',
    '✔️': '[OK]',
    '🔧': '[TOOL]',
    '📥': '[DOWNLOAD]',
    '🔄': '[RETRY]',
    '✓': '[OK]',
    'ℹ️': '[INFO]',
    '💡': '[TIP]',
    '📋': '[NOTE]',
    '🎬': '[VIDEO]',
    '📅': '[DATE]',
    '🔍': '[SEARCH]',
    '⏱️': '[TIMER]',
    '🏁': '[FINISH]',
    '📊': '[STATS]',
    '🧠': '[AI]',
    '💡': '[TIP]'
}

for emoji, text in replacements.items():
    content = content.replace(emoji, text)

with open('video_orientation_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Emojis replaced with plain text')