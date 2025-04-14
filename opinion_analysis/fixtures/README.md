# Build your vectore store

This service is currently only support `.md` file with structured headers.(header 1, header 2,...)  
Please place your markdwon file under this directory before running the service.

```bash
fixtures
├── README.md
├── <your markdown file>
├── ...
└── ...
```

The vectorstore directory will be saved here when building up the service, noted that since this is only for a MVP demo requirement, it is not supported for multiple vecterstore index management,that is, the service will only build and read the specific directory of vectorstore files.

If you tend to modify the vectorstore, please stop the service and then remove the vectorstore directory, and restart the service.
