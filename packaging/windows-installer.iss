#ifndef StageDir
  #error StageDir is required
#endif
#ifndef WorkerVersion
  #error WorkerVersion is required
#endif
#ifndef ReleaseKey
  #error ReleaseKey is required
#endif

[Setup]
AppId={{C46A4328-55A7-46B1-A277-5FC4DC63BC91}
AppName=MARP Inference Worker
AppVersion={#WorkerVersion}
DefaultDirName={localappdata}\MARP\Worker
DefaultGroupName=MARP Inference Worker
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#StageDir}\installer
OutputBaseFilename=marp-inference-worker-{#WorkerVersion}-windows-x64-setup
Compression=lzma2/ultra64
SolidCompression=yes
Uninstallable=yes

[Dirs]
Name: "{app}\state"; Flags: uninsneveruninstall

[Files]
Source: "{#StageDir}\launcher\marp-worker-launcher.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\versions\{#ReleaseKey}\*"; DestDir: "{app}\versions\{#ReleaseKey}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MARP Worker (background)"; Filename: "{app}\marp-worker-launcher.exe"; Parameters: "--screen off"
Name: "{group}\MARP Worker (window)"; Filename: "{app}\marp-worker-launcher.exe"; Parameters: "--screen window"
Name: "{group}\MARP Worker (fullscreen)"; Filename: "{app}\marp-worker-launcher.exe"; Parameters: "--screen fullscreen"
Name: "{group}\Finish current MARP work, then pause"; Filename: "{app}\marp-worker-launcher.exe"; Parameters: "--control finish"
Name: "{group}\Stop MARP work now, then pause"; Filename: "{app}\marp-worker-launcher.exe"; Parameters: "--control stop"
Name: "{group}\Resume MARP work"; Filename: "{app}\marp-worker-launcher.exe"; Parameters: "--control resume"
Name: "{userstartup}\MARP Inference Worker"; Filename: "{app}\marp-worker-launcher.exe"; Parameters: "--screen off"

[Run]
Filename: "{app}\versions\{#ReleaseKey}\marp-worker.exe"; Parameters: "--activate-code-file &quot;{tmp}\marp-worker-activation.txt&quot; --coordinator-url &quot;{code:GetCoordinatorUrl}&quot; --state-dir &quot;{app}\state&quot;"; StatusMsg: "Activating this worker..."; Flags: runhidden waituntilterminated
Filename: "{app}\marp-worker-launcher.exe"; Parameters: "--screen off"; Description: "Start the MARP worker"; Flags: nowait postinstall skipifsilent

[Code]
var
  ConnectPage: TInputQueryWizardPage;

procedure InitializeWizard;
begin
  ConnectPage := CreateInputQueryPage(wpSelectDir,
    'Connect this worker to MARP',
    'Enter the MARP API address and the one-time activation code.',
    'The activation code is exchanged once for a credential protected for this Windows user.');
  ConnectPage.Add('MARP API URL:', False);
  ConnectPage.Add('Activation code:', True);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = ConnectPage.ID then
  begin
    if (Trim(ConnectPage.Values[0]) = '') or (Trim(ConnectPage.Values[1]) = '') then
    begin
      MsgBox('Both the MARP API URL and activation code are required.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

function GetCoordinatorUrl(Param: String): String;
begin
  Result := Trim(ConnectPage.Values[0]);
end;

function JsonEscape(Value: String): String;
begin
  Result := Value;
  StringChangeEx(Result, '\', '\\', True);
  StringChangeEx(Result, '"', '\"', True);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Config: String;
begin
  if CurStep = ssPostInstall then
  begin
    Config := '{"coordinator_url":"' + JsonEscape(Trim(ConnectPage.Values[0])) + '","screen":"off","api_port":8010}';
    SaveStringToFile(ExpandConstant('{app}\config.json'), Config, False);
    SaveStringToFile(ExpandConstant('{app}\active.json'), '{"release":"{#ReleaseKey}"}', False);
    SaveStringToFile(ExpandConstant('{tmp}\marp-worker-activation.txt'), Trim(ConnectPage.Values[1]), False);
  end;
end;
