from setuptools import setup


setup(
	options={
		"build_apps": {
			"gui_apps": {
				"ss-battle": "main.py",
			},
			"log_filename": "$USER_APPDATA/SoulSymphony2/output.log",
			"log_append": False,
			"include_patterns": [
				"README.md",
				"models/**",
				"graphics/**",
				"soundfx/**",
				"bgm/**",
				"docs/**",
			],
			"exclude_patterns": [
				"**/__pycache__/**",
				"**/*.pyc",
				"**/*.pyo",
				"webbuild/**",
				"potential_tracks/**",
				"scripts/**",
			],
			"exclude_modules": [
				"_bootlocale",
				"_posixsubprocess",
				"grp",
			],
			"plugins": [
				"pandagl",
				"p3openal_audio",
				"p3ffmpeg",
			],
			"platforms": [
				"manylinux2014_x86_64",
				"macosx_10_9_x86_64",
				"win_amd64",
			],
		}
	}
)
